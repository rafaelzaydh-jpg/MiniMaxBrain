"""Assistente automático e amigável para conversão de GGUF para MiniMaxBrain (MMB)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Configura stdout/stderr para UTF-8 seguro no Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from minimaxbrain.config import load_external_bundle
from minimaxbrain.gguf import gguf_summary, load_gguf
from minimaxbrain.gguf_moe import pack_moe_gguf
from minimaxbrain.storage import create_model_seal
from minimaxbrain.units import format_bytes


def banner() -> None:
    print("=" * 65)
    print("       [*] MiniMaxBrain (MMB) - Conversor Automatico de Modelos      ")
    print("=" * 65)
    print()


def main() -> int:
    banner()
    conversor_dir = ROOT_DIR / "conversor"
    conversor_dir.mkdir(exist_ok=True)

    # Search for .gguf files in conversor/
    gguf_files = list(conversor_dir.glob("*.gguf")) + list(conversor_dir.glob("**/*.gguf"))
    gguf_files = sorted(set(gguf_files))

    if not gguf_files:
        print("[!] Nenhum arquivo '.gguf' foi encontrado na pasta 'conversor/'!\n")
        print("Como usar:")
        print(f"   1. Copie seu modelo (ex: 'modelo.gguf') para dentro da pasta:")
        print(f"      {conversor_dir.resolve()}\n")
        print("   2. Execute o 'conversor.bat' novamente.\n")
        return 1

    selected_gguf = None
    if len(gguf_files) == 1:
        selected_gguf = gguf_files[0]
        print(f"[*] Modelo detectado automaticamente:")
        print(f"   Arquivo: {selected_gguf.name} ({format_bytes(selected_gguf.stat().st_size)})\n")
    else:
        print(f"[+] Foram encontrados {len(gguf_files)} modelos na pasta 'conversor/':\n")
        for i, f in enumerate(gguf_files, 1):
            print(f"   [{i}] {f.name} ({format_bytes(f.stat().st_size)})")
        print()
        while True:
            choice = input(f"Selecione o numero do modelo (1-{len(gguf_files)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(gguf_files):
                selected_gguf = gguf_files[int(choice) - 1]
                break
            print("Opcao invalida, tente novamente.")

    print("=" * 65)
    print("[1/4] Inspecionando a estrutura do modelo GGUF...")
    try:
        model = load_gguf(selected_gguf)
        summary = gguf_summary(model)
        arch = summary.get("architecture", "desconhecida")
        tensors = summary.get("tensor_count", 0)
        print(f"      ✔ Arquitetura: {arch}")
        print(f"      ✔ Total de Tensores: {tensors}")
    except Exception as exc:
        print(f"      [ERRO] Erro ao ler GGUF: {exc}")
        return 1

    # Define output directory
    clean_stem = selected_gguf.stem.lower().replace(" ", "-").replace("_", "-")
    output_dir = conversor_dir / f"{clean_stem}-mmbw"

    if output_dir.exists():
        print(f"\n[!] A pasta de saida ja existe: {output_dir.name}")
        ans = input("Deseja sobrescrever/recriar? (S/N): ").strip().upper()
        if ans not in ("S", "SIM", "Y", "YES"):
            print("Operacao cancelada pelo usuario.")
            return 0
        import shutil
        shutil.rmtree(output_dir)

    print("\n[2/4] Fatiando e empacotando especialistas para o formato MMBW...")
    print("      (Isso pode levar alguns minutos dependendo do tamanho do arquivo...)")
    try:
        manifest_path = pack_moe_gguf(selected_gguf, output_dir)
        print(f"      ✔ Pesos empacotados com sucesso em: {output_dir.name}")
    except Exception as exc:
        print(f"      [ERRO] Erro durante o empacotamento: {exc}")
        return 1

    print("\n[3/4] Criando arquivo de configuracao modular (gate.json)...")
    config_data = {
        "schema_version": "mmb-external-gate-config-v1",
        "model_map": "model.mmb-map.json",
        "memory": {
            "ram_budget": "4GiB",
            "resident_experts": null,
            "kv_cache": "512MiB",
            "scratch": "256MiB",
            "transport": "heap",
            "lease_timeout_seconds": 120
        },
        "io": {
            "workers": 2,
            "prefetch_queue": 32,
            "integrity": "seal"
        },
        "server": {
            "host": "127.0.0.1",
            "port": 55321
        },
        "telemetry": {
            "enabled": true
        },
        "model_memory": {
            "enabled": false,
            "path": "state/model-memory.sqlite3"
        }
    }
    config_path = output_dir / "gate.json"
    config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    print(f"      ✔ Configuracao salva: {config_path.name}")

    print("\n[4/4] Gerando Selo Criptografico de Integridade Pre-Voo (mmb seal)...")
    try:
        config, model_map = load_external_bundle(config_path)
        seal_data = create_model_seal(model_map)
        print(f"      ✔ Selo gravado com sucesso: model.verified.json")
        print(f"      ✔ Total de {seal_data['total_blocks']} blocos verificados!")
    except Exception as exc:
        print(f"      [AVISO] Nao foi possivel gerar o selo automaticamente: {exc}")

    print("\n" + "=" * 65)
    print("CONVERSAO CONCLUIDA COM SUCESSO!")
    print("=" * 65)
    print(f"Pasta do modelo pronto: {output_dir.resolve()}\n")
    print("Para iniciar o servico do Gate com este modelo:")
    print(f"   python mmb.py serve --config {config_path.relative_to(ROOT_DIR)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
