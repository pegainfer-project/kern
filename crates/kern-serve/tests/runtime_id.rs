//! The handover contract with `kern server`: the launcher hands over the
//! runtime it was built with, and a server built from a different one refuses
//! to serve, because it would serve the runtime from before the change the
//! launcher is testing.

/// The variable `kern server` sets before it execs the server.
const LAUNCHER: &str = kern_runtime::ID_VAR;

fn serve_with(id: Option<&str>) -> std::process::Output {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_kern-serve"));
    command.arg("--version");
    if let Some(id) = id {
        command.env(LAUNCHER, id);
    }
    command.output().unwrap()
}

#[test]
fn a_server_of_another_runtime_refuses_to_serve() {
    let out = serve_with(Some("0000000000ff"));
    assert!(!out.status.success(), "a mismatched runtime served anyway");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("0000000000ff") && err.contains(kern_runtime::ID), "{err}");
    assert!(err.contains("crates/kern-serve/Cargo.toml"), "the error names no fix: {err}");
}

#[test]
fn the_same_runtime_serves_and_a_direct_launch_is_left_alone() {
    for id in [Some(kern_runtime::ID), None] {
        let out = serve_with(id);
        assert!(out.status.success(), "{id:?}: {}", String::from_utf8_lossy(&out.stderr));
        let reported = String::from_utf8_lossy(&out.stdout);
        assert!(reported.contains(&format!("(runtime {})", kern_runtime::ID)), "{id:?}: {reported}");
    }
}
