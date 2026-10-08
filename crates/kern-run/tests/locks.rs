//! The two workspaces ship one runtime. `crates/kern-serve` resolves its own
//! graph, so a dependency bumped in one lockfile and not the other would give
//! `kern` and `kern-serve` different versions of the same crate — and the
//! runtime id cannot see it, because what it digests is sources.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use toml::Value;

/// The crates both binaries link, the ones `kern-runtime`'s build script
/// digests.
const SHARED: [&str; 4] = ["kern-manifest", "kern-pool", "kern-runtime", "kern-run"];

/// A package by name and version.
type Package = (String, String);

/// The lockfile: every version of every name, and each package's dependencies
/// as the lock spells them (`name` or `name version`).
struct Lock {
    versions: BTreeMap<String, Vec<String>>,
    edges: BTreeMap<Package, Vec<String>>,
}

impl Lock {
    fn read(path: &Path) -> Lock {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let parsed: Value = toml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut versions: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut edges: BTreeMap<Package, Vec<String>> = BTreeMap::new();
        for package in parsed["package"].as_array().expect("a [[package]] table") {
            let name = package["name"].as_str().expect("a package name").to_string();
            let version = package["version"].as_str().expect("a package version").to_string();
            versions.entry(name.clone()).or_default().push(version.clone());
            let deps = package.get("dependencies").and_then(Value::as_array).cloned().unwrap_or_default();
            edges.insert(
                (name, version),
                deps.iter().map(|d| d.as_str().expect("a dependency as a string").to_string()).collect(),
            );
        }
        Lock { versions, edges }
    }

    fn packages(&self) -> BTreeSet<Package> {
        self.edges.keys().cloned().collect()
    }

    /// Every package reachable from the shared crates. A shared crate's
    /// dev-dependencies are skipped: they are in the runtime's lock and are
    /// not resolved at all in the serving one, and nothing the binaries link
    /// needs them.
    fn closure(&self, repo: &Path) -> BTreeSet<Package> {
        let mut seen = BTreeSet::new();
        let mut queue: Vec<Package> = SHARED
            .iter()
            .flat_map(|name| self.versions.get(*name).into_iter().flatten().map(|v| (name.to_string(), v.clone())))
            .collect();
        while let Some(package) = queue.pop() {
            if !seen.insert(package.clone()) {
                continue;
            }
            let skip = if SHARED.contains(&package.0.as_str()) { dev_only(repo, &package.0) } else { BTreeSet::new() };
            for dep in &self.edges[&package] {
                let (name, version) = match dep.rsplit_once(' ') {
                    Some((name, version)) => (name.to_string(), Some(version.to_string())),
                    None => (dep.clone(), None),
                };
                if skip.contains(&name) {
                    continue;
                }
                match version {
                    Some(version) => queue.push((name, version)),
                    None => {
                        queue.extend(self.versions.get(&name).into_iter().flatten().map(|v| (name.clone(), v.clone())))
                    }
                }
            }
        }
        seen
    }
}

/// The dev-dependencies a crate declares and does not also use for real.
fn dev_only(repo: &Path, krate: &str) -> BTreeSet<String> {
    let path = repo.join("crates").join(krate).join("Cargo.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let parsed: Value = toml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let names = |table: &str| -> BTreeSet<String> {
        parsed.get(table).and_then(Value::as_table).map(|t| t.keys().cloned().collect()).unwrap_or_default()
    };
    let mut dev = names("dev-dependencies");
    dev.retain(|name| !names("dependencies").contains(name) && !names("build-dependencies").contains(name));
    dev
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("the repository root")
        .to_path_buf()
}

#[test]
fn both_workspaces_resolve_the_runtime_at_the_same_versions() {
    let repo = repo();
    let runtime = Lock::read(&repo.join("Cargo.lock"));
    let serve = Lock::read(&repo.join("crates").join("kern-serve").join("Cargo.lock"));
    let closure = runtime.closure(&repo);
    let resolved = serve.packages();
    let missing: Vec<&Package> = closure.difference(&resolved).collect();
    assert!(
        missing.is_empty(),
        "the serving workspace does not resolve {} of the runtime's {} packages at the same version: {:?}",
        missing.len(),
        closure.len(),
        &missing[..missing.len().min(8)]
    );
}
