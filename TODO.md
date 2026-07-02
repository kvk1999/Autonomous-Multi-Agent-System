# TODO

- [x] Make Streamlit dark-mode deterministic in `app.py` (apply `dark-mode` class + ensure injected CSS variables are used consistently).
- [x] Port minimal deck-like card styling into Streamlit by adding scoped classes to `styles.css` (no breaking changes to existing tokens).
- [x] Add defensive guards + better error handling in `app.py` for:
  - route optimization execution
  - Gemini query path
  - any UI sections that assume session state.
- [ ] Run `streamlit run app.py` (or `run_app.ps1` / `run_app.cmd`) and visually verify styling + that dispatch works without crashing.
- [x] Fix dark-mode placeholder/text colors so placeholder/background appear black as required.



