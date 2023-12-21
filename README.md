# AI Student for Negative Keywords

## Deploy

If not already done, create a [Consent Screen](https://console.cloud.google.com/apis/credentials/consent) on the GCP Console and publish it.

Then create a new [OAuth 2.0 Client ID](https://pantheon.corp.google.com/apis/credentials) with "Web Application" as type (leave the rest blank). Take note of the Client ID and Client Secret, you are going to need them in the next step.

Open up Cloud Shell:

    git clone .../neg-keywords-cleaner.git
    cd neg-keywords-cleaner

    echo 'google_oauth_client_id = "..."' > terraform/secrets.tfvars
    echo 'google_oauth_client_secret = "..."' >> terraform/secrets.tfvars

    export TF_VAR_project_id=$(gcloud config list --format 'value(core.project)')

You can edit the ```terraform/main.tf``` file under the google_iam_policy.noauth section, to restrict access to specific users (the app will be visible to all users with the url per default).

    # Build and deploy container
    docker build -t gcr.io/$TF_VAR_project_id/negatives:v1 . --no-cache
    docker push gcr.io/$TF_VAR_project_id/negatives:v1

If it doesn't deploy, you might have to run `gcloud auth configure-docker` and try again.

    # Deploys the solution.
    cd terraform/
    terraform init -upgrade
    terraform apply -var-file secrets.tfvars

    > App URL: https://myapp.run.app

Go back to Credentials > OAuth 2.0 Client ID and select your Client ID.
Add the App URL into the list of Authorized Redirect URL.


## Uninstall

You can remove all GCP resources with one command:

    terraform destroy

## Run the app on a local machine

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
    (.venv) streamlit run --server.headless=false src/AI_student.py
