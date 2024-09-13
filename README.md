# Negative Keyword Cleaner

### This is NOT an official Google product

Negative Keyword Cleaner is a solution that leverages Large Language Models to
identify and help you remove Negative Keywords blocking traffic from your
Google Ads account.

## Prerequisites

1.
A [Google Ads Developer token](https://developers.google.com/google-ads/api/docs/first-call/dev-token#:~:text=A%20developer%20token%20from%20Google,SETTINGS%20%3E%20SETUP%20%3E%20API%20Center.)
1. A [Google Cloud Project](https://cloud.google.com/) with billing attached
1.
A [Consent Screen](https://console.cloud.google.com/apis/credentials/consent)
on the GCP Console
1.
Create [OAuth2 Credentials](https://developers.google.com/identity/protocols/oauth2#1.-obtain-oauth-2.0-credentials-from-the-dynamic_data.setvar.console_name-.)
of type **Web**

## Deployment

To avoid potential conflicts, we recommend deploying this solution in a new
Google Cloud Platform (GCP) project. Existing GCP projects may already be
linked to a specific Google Ads Manager Account (MCC) through prior use of the
Google Ads API.

If you're certain that your chosen GCP project has not been used with the
Google Ads API, or that it has only been used with the same MCC you intend to
use now, you may proceed with the steps below.

1. If not already done, create
   a [Consent Screen](https://console.cloud.google.com/apis/credentials/consent)
   on the GCP Console and publish it.
2. Create a
   new [OAuth 2.0 Client ID](https://console.cloud.google.com/apis/credentials)
   with **Web Application** as type (leave the rest blank).
    * Take note of the **Client ID** and **Client Secret**, you are going to
      need them later.
3. In APIs & Service, enable **Artifact Registry API**
4. In Cloud Shell
   ```
   git clone https://professional-services.googlesource.com/solutions/ai_student_for_negative_keywords
   cd ai_student_for_negative_keywords
   ```
5. Replace the ... with the Client ID and Client Secret obtained in Step 2
   ```
   echo 'google_oauth_client_id = "..."' > terraform/secrets.tfvars
   echo 'google_oauth_client_secret = "..."' >> terraform/secrets.tfvars
   echo 'mcc_id = "XXXXXXXXXX"' >> terraform/secrets.tfvars
   echo 'google_ads_api_token = "..."' >> terraform/secrets.tfvars
   echo 'open_ai_key = "..."' >> terraform/secrets.tfvars
   export TF_VAR_project_id=$(gcloud config list --format 'value(core.project)')
   ```
   > You can edit the ```terraform/main.tf``` file under the
   google_iam_policy.noauth section, to restrict access to specific users (the
   app
   will be visible to all users with the url per default).

6. Build and deploy container
   ```
   docker build -t gcr.io/$TF_VAR_project_id/negatives:v1 . --no-cache
   docker push gcr.io/$TF_VAR_project_id/negatives:v1
   ```
   > If it doesn't deploy, you might have to run `gcloud auth configure-docker`
   and
   try again.

7. Deploy the solution
   ```
   cd terraform/
   terraform init -upgrade
   terraform apply -var-file secrets.tfvars
   ```
   > Take note of the **App URL**. *Example: https://myapp.run.app*

8. Go back to Credentials > OAuth 2.0 Client ID and select your Client ID.
    * Add the App URL into the list of **Authorized Redirect URL**.

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

## Disclaimer

**This is not an officially supported Google product.**

*Copyright 2024 Google LLC. Supported by Google LLC and/or its affiliate(s).
This solution, including any related sample code or data, is made available on
an “as is,” “as available,” and “with all faults” basis, solely for
illustrative purposes, and without warranty or representation of any kind. This
solution is experimental, unsupported and provided solely for your convenience.
Your use of it is subject to your agreements with Google, as applicable, and
may constitute a beta feature as defined under those agreements. To the extent
that you make any data available to Google in connection with your use of the
solution, you represent and warrant that you have all necessary and appropriate
rights, consents and permissions to permit Google to use and process that data.
By using any portion of this solution, you acknowledge, assume and accept all
risks, known and unknown, associated with its usage and any processing of data
by Google, including with respect to your deployment of any portion of this
solution in your systems, or usage in connection with your business, if at all.
With respect to the entrustment of personal information to Google, you will
verify that the established system is sufficient by checking Google's privacy
policy and other public information, and you agree that no further information
will be provided by Google.*
