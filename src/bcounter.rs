use crate::pncounter;
use crate::traits::{Causal, CmRDT, CvRDT};
use crate::{Actor, Dot, Error, Map, PNCounter, VClock};

use num_bigint::BigInt;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, LinkedList};

/// 'BCounter' is a counter with a bounded condition on the inner value.
#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct BCounter<A: Actor> {
    /// The underlying PNCounter
    inner: PNCounter<A>,

    /// The lower bound, TODO: make adjustable
    lb: u64,

    /// Handling state is easier for transfers matrix
    transfers: BTreeMap<(A, A), u64>,
}

/// An op that wraps the internal pncounter op
/// Ship these ops to other replicas to have them sync up.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Op<A: Actor> {
    /// The op for the pncounter
    pub inner: pncounter::Op<A>,
}

impl<A: Actor> Default for BCounter<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Actor> CmRDT for BCounter<A> {
    type Op = Op<A>;

    fn apply(&mut self, op: Self::Op) {
        self.inner.apply(op.inner)
    }
}

impl<A: Actor> CvRDT for BCounter<A> {
    fn merge(&mut self, other: Self) {
        // Merge the inner PNCounter
        self.inner.merge(other.inner);

        // Apply Max Operation to the table
        // TODO: improve efficiency here, too lazy right now
        for ((other_tx, other_rx), other_v) in other.transfers.iter() {
            match self.transfers.get(&(other_tx.clone(), other_rx.clone())) {
                Some(self_v) => {
                    if self_v < other_v {
                        self.transfers
                            .insert((other_tx.clone(), other_rx.clone()), other_v.clone());
                    }
                }
                None => {
                    self.transfers
                        .insert((other_tx.clone(), other_rx.clone()), other_v.clone());
                }
            }
        }
    }
}

impl<A: Actor> BCounter<A> {
    /// Produce a new 'BCounter'
    pub fn new() -> Self {
        Self {
            inner: PNCounter::new(),
            lb: 0,
            transfers: BTreeMap::new(),
        }
    }

    /// Generate an Op to increment the counter
    pub fn inc(&self, actor: A) -> Op<A> {
        Op {
            inner: self.inner.inc(actor),
        }
    }

    /// Generate an Op to decrement the counter.
    pub fn dec(&self, actor: A) -> Result<Op<A>, Error> {
        match self.quota(actor.clone()) > BigInt::from(0) {
            true => Ok(Op {
                inner: self.inner.dec(actor.clone()),
            }),
            false => Err(Error::OperationNotAllowed(
                "Local quota will be exceeded".into(),
            )),
        }
    }

    /// Return the current value of this counter (P-N).
    pub fn read(&self) -> BigInt {
        self.inner.read()
    }

    /// Return the local quota avaialble to this node
    /// TODO: make private?
    pub fn quota(&self, actor: A) -> BigInt {
        let mut out = self.inner.read();
        for ((tx, rx), v) in self
            .transfers
            .iter()
            .filter(|((tx, rx), _)| *tx == actor || *rx == actor)
        {
            match (*tx == actor, *rx == actor) {
                (true, false) => out = out - v, // If we are the sender, subtract
                (false, true) => out = out + v, // If we are the receiver, add
                _ => {}
            }
        }
        BigInt::from(out)
    }

    /// Transfers quota between actors
    pub fn transfer(&mut self, tx_actor: A, rx_actor: A, amount: u64) -> Result<(), Error> {
        if self.quota(tx_actor.clone()) >= BigInt::from(amount) {
            match self.transfers.get(&(tx_actor.clone(), rx_actor.clone())) {
                Some(val) => {
                    self.transfers
                        .insert((tx_actor.clone(), rx_actor.clone()), val + amount);
                }
                None => {
                    self.transfers
                        .insert((tx_actor.clone(), rx_actor.clone()), amount);
                }
            };
            return Ok(());
        }
        Err(Error::OperationNotAllowed(
            "cannot transfer more than local quota".into(),
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use std::collections::BTreeSet;

    use quickcheck::quickcheck;

    const ACTOR_MAX: u8 = 11;

    fn build_op(prims: (u8, u64, bool)) -> Op<u8> {
        let (actor, counter, dir_choice) = prims;
        Op {
            inner: pncounter::Op {
                dot: Dot { actor, counter },
                dir: if dir_choice {
                    pncounter::Dir::Pos
                } else {
                    pncounter::Dir::Neg
                },
            },
        }
    }

    quickcheck! {
        fn prop_merge_converges(op_prims: Vec<(u8, u64, bool)>) -> bool {
            let ops: Vec<Op<u8>> = op_prims.into_iter().map(build_op).collect();

            let mut results = BTreeSet::new();

            // Permute the interleaving of operations should converge.
            // Largely taken directly from orswot
            for i in 2..ACTOR_MAX {
                let mut witnesses: Vec<BCounter<u8>> =
                    (0..i).map(|_| BCounter::new()).collect();
                for op in ops.iter() {
                    let index = op.inner.dot.actor as usize % i as usize;
                    let witness = &mut witnesses[index];
                    witness.apply(op.clone());
                }
                let mut merged = BCounter::new();
                for witness in witnesses.iter() {
                    merged.merge(witness.clone());
                }

                results.insert(merged.read());
                if results.len() > 1 {
                    println!("opvec: {:?}", ops);
                    println!("results: {:?}", results);
                    println!("witnesses: {:?}", &witnesses);
                    println!("merged: {:?}", merged);
                }
            }
            results.len() == 1
        }
    }

    #[test]
    fn test_basic() {
        let mut sut_1 = BCounter::new();
        let mut sut_2 = BCounter::new();
        assert_eq!(sut_1.read(), 0.into());
        assert_eq!(sut_2.read(), 0.into());
        assert!(sut_1.dec("A").is_err());

        sut_1.apply(sut_1.inc("A"));
        sut_1.apply(sut_1.inc("A"));
        sut_2.apply(sut_2.inc("B"));
        assert_eq!(sut_1.read(), 2.into());
        assert_eq!(sut_1.quota("A"), 2.into());
        assert_eq!(sut_2.read(), 1.into());
        assert_eq!(sut_2.quota("B"), 1.into());

        sut_2.transfer("B", "A", 1).expect("should not fail");
        assert_eq!(sut_2.quota("B"), 0.into());
        assert_eq!(sut_1.quota("A"), 2.into());

        sut_2.merge(sut_1.clone());
        sut_1.merge(sut_2.clone());
        assert_eq!(sut_1.quota("A"), 4.into());
        assert_eq!(sut_2.quota("B"), 2.into());
        assert_eq!(sut_1, sut_2);

        let mut done = false;
        while !done {
            let op = sut_1.dec("A");
            if op.is_err() {
                break;
            }
            sut_1.apply(op.unwrap());
        }
        done = false;
        while !done {
            let op = sut_2.dec("B");
            if op.is_err() {
                break;
            }
            sut_2.apply(op.unwrap());
        }
        assert_eq!(sut_1.quota("A"), 0.into());
        assert_eq!(sut_2.quota("B"), 0.into());
        assert_eq!(sut_1.read(), (-1).into());
        assert_eq!(sut_2.read(), (1).into());
    }
}
