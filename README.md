# AI Student for Negative Keywords

## Run the app

Create a test client for local development:

    # Create the OAuth Client (from the Google Cloud Platform UI, cannot be done programmatically)
    open https://console.cloud.google.com

    # Copy the OAuth2.0 Client Id and Client Secret in `.env`
    echo "OAUTH_CLIENT_ID=xxx" > .env
    echo "OAUTH_CLIENT_SECRET=yyy" >> .env
    echo "OAUTH_REDIRECT_URI=http://localhost:8080" >> .env

Using a standard virtualenv:

    # (Optional) if you have pyenv
    pyenv update
    pyenv install 3.11.3
    pyenv local 3.11.3
    
    # Installs dependencies
    python -m venv .venv
    source .venv/bin/activate
    (.venv) pip install -r requirements.txt
    (.venv) pip install python-dotenv

    # Runs the streamlit server
    (.venv) streamlit run src/AI_student.py

## Deploy

Open CloudShell on your GCP

    git clone .../neg-keywords-cleaner.git
    cd neg-keywords-cleaner/terraform/

    export TF_VAR_project_id=$(gcloud config list --format 'value(core.project)')
    export TF_VAR_google_oauth_client_id=...
    export TF_VAR_google_oauth_client_secret=...

    # Deploys the solution.
    terraform init -upgrade
    terraform apply

    > App URL: https://myapp.appspot.com

## Uninstall

You can remove all GCP resources with one command:

    terraform destroy
