# LaPaHV — Análise de Parasitoses (Streamlit)

App Streamlit para o Laboratório de Parasitologia Humana e Veterinária: baixa um modelo de
planilha, recebe o arquivo preenchido e devolve a análise epidemiológica completa (prevalência
por criança, por espécie, poliparasitismo, comparação de métodos diagnósticos etc.), com
exportação em Excel.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre em `http://localhost:8501`.

## Publicar no Streamlit Community Cloud (grátis)

1. Suba esta pasta (`app.py`, `analysis_engine.py`, `logo.png`, `requirements.txt`) para um
   repositório no GitHub.
2. Acesse [share.streamlit.io](https://share.streamlit.io) e faça login com sua conta GitHub.
3. Clique em **"New app"**, escolha o repositório, a branch e aponte o **main file path** para
   `app.py`.
4. Clique em **Deploy** — em um ou dois minutos o app fica no ar em um link tipo
   `https://seu-app.streamlit.app`.

Qualquer novo `git push` no repositório atualiza o app publicado automaticamente.

## Estrutura

- `app.py` — interface Streamlit (upload, métricas, gráficos, exportação)
- `analysis_engine.py` — motor de análise (parsing de resultados, agregação por criança,
  cálculo de prevalências) — mesma lógica usada no notebook e no site em HTML
- `logo.png` — logo do LaPaHV com fundo transparente
- `requirements.txt` — dependências Python
