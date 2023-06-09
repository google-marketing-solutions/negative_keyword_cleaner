# AI Student for Negative Keywords

## Run the app

    # (Optional) if you have pyenv
    pyenv update
    pyenv install 3.11.3
    pyenv local 3.11.3
    
    # Installs dependencies
    python -m venv .venv
    source .venv/bin/activate
    (.venv) pip install -r requirements.txt

    # Runs the streamlit server
    (.venv) streamlit run src/AI_student.py
