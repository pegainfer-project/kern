//! The peers' handshake on the wire: an offer survives its bytes whole,
//! anything short, long or foreign is refused rather than misread, and a
//! handle only this process can map has no wire form and says why.

use std::collections::BTreeMap;

use kern_runtime::{Offer, PeerHandle};

fn handle(i: usize) -> PeerHandle {
    let mut fabric = [0u8; 64];
    fabric.iter_mut().enumerate().for_each(|(j, b)| *b = (i * 31 + j * 7) as u8);
    PeerHandle::Fabric { fabric, bytes: (i as u64 + 1) << 21 }
}

fn offer(handles: usize, groups: usize, name: &str) -> Offer {
    Offer {
        manifest: 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(handles as u64 + 1),
        handles: (0..handles).map(|i| (format!("{name}{i}"), handle(i))).collect(),
        nccl: (0..groups).map(|g| (format!("g{g}"), [g as u8 ^ 0x5a; 128])).collect(),
    }
}

fn offers() -> Vec<Offer> {
    let names = ["", "tp_ar_lamport", "dcp.lamport/é"];
    (0..4).flat_map(|h| (0..3).flat_map(move |g| names.map(|n| offer(h, g, n)))).collect()
}

#[test]
fn an_offer_reads_back_as_written() {
    for o in offers() {
        let b = o.to_bytes().unwrap();
        assert_eq!(
            (Offer::from_bytes(&b).unwrap(), b.clone()),
            (o.clone(), Offer::from_bytes(&b).unwrap().to_bytes().unwrap())
        );
    }
}

#[test]
fn a_cut_padded_or_foreign_offer_is_refused() {
    for o in offers() {
        let b = o.to_bytes().unwrap();
        for n in 0..b.len() {
            assert!(Offer::from_bytes(&b[..n]).is_err(), "{n} of {} bytes", b.len());
        }
        let mut long = b.clone();
        long.push(0);
        let mut foreign = b.clone();
        foreign[0] ^= 1;
        assert_eq!((Offer::from_bytes(&long).is_err(), Offer::from_bytes(&foreign).is_err()), (true, true));
    }
}

#[test]
fn a_local_handle_has_no_wire_form_and_names_the_imex_channel() {
    let o = Offer {
        manifest: 1,
        handles: BTreeMap::from([("lamport".to_string(), PeerHandle::Local { handle: 7, bytes: 1 << 21 })]),
        nccl: BTreeMap::new(),
    };
    let e = o.to_bytes().unwrap_err().to_string();
    assert_eq!((e.contains("`lamport`"), e.contains("imex-channels")), (true, true), "{e}");
}
