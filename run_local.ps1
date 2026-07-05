$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

python -m pip install -r requirements.txt

try {
    ollama list | Out-Null
} catch {
    Write-Host "Ollama is not responding. Start it with: ollama serve" -ForegroundColor Yellow
}

python app_gradio.py
