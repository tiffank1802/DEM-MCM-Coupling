# Multi-stage Dockerfile for DEM-MCM Coupling (pure-Rust 3-D Studio + CLI)
FROM rust:1.88-slim-bookworm AS builder

WORKDIR /build
COPY Cargo.toml ./
COPY src ./src
COPY tests_rs ./tests_rs

RUN cargo build --release --bin dem-mcm-gui --bin dem-mcm-sweep

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/dem-mcm-gui /usr/local/bin/dem-mcm-gui
COPY --from=builder /build/target/release/dem-mcm-sweep /usr/local/bin/dem-mcm-sweep
COPY data/chunks ./data/chunks

ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD ["/bin/bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/8080 && printf 'GET /api/health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n' >&3 && grep -q '200 OK' <&3"]

ENTRYPOINT ["/usr/local/bin/dem-mcm-gui"]
CMD ["--port", "8080", "--data-dir", "/app/data/chunks"]
