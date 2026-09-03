//! Emit the manifest wire-format definition as JSON Schema.
//!
//! The Rust types in `kern_manifest::types` are the ground truth (the parser
//! is the law); this is their publishable projection. The committed copy at
//! `schema/manifest-v4.schema.json` is golden-checked in CI:
//!
//! ```sh
//! cargo run -p kern-manifest --example gen_schema > schema/manifest-v4.schema.json
//! ```

fn main() {
    let schema = schemars::schema_for!(kern_manifest::Manifest);
    let mut v = serde_json::to_value(&schema).expect("schema serializes");
    v["$id"] = "https://kern-baa.pages.dev/schema/manifest-v4.schema.json".into();
    v["title"] = "kern manifest v4".into();
    println!("{}", serde_json::to_string_pretty(&v).expect("schema serializes"));
}
