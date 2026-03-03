fn main() {
    if std::env::var_os("ORT_LIB_LOCATION").is_none() {
        println!(
            "cargo:warning=ORT_LIB_LOCATION is not set. \
             ort-sys may download a prebuilt ONNX Runtime binary from an external URL. \
             Set ORT_LIB_LOCATION to point to a locally built ONNX Runtime to avoid this."
        );
    }
}
