# Negative Keyword Cleaner

### This is NOT an official Google product

Negative Keyword Cleaner is a solution that leverages Large Language Models to
identify and help you remove Negative Keywords blocking traffic from your
Google Ads account.

## Prerequisites

1. A [Google Ads Developer token](https://developers.google.com/google-ads/api/docs/first-call/dev-token#:~:text=A%20developer%20token%20from%20Google,SETTINGS%20%3E%20SETUP%20%3E%20API%20Center.)
2. A [Consent Screen](https://console.cloud.google.com/apis/credentials/consent) on the GCP Console
3. Create [OAuth2 Credentials](https://developers.google.com/identity/protocols/oauth2#1.-obtain-oauth-2.0-credentials-from-the-dynamic_data.setvar.console_name-.)
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
3. In APIs & Service, enable **Artifact Registry API**, **Cloud Run API**, and **IAM API**.
4. In Cloud Shell
   ```
   git clone https://github.com/google-marketing-solutions/negative_keyword_cleaner
   cd negative_keyword_cleaner
   export TF_VAR_project_id=$(gcloud config list --format 'value(core.project)')
   ```
5. Create a service account for the Cloud Run service to use.
   ```
   gcloud iam service-accounts create neg-keywords-cleaner \
     --display-name "Negative Keywords Cleaner"
   ```
6. Grant the necessary permissions to the service account.
    ```
    gcloud projects add-iam-policy-binding $TF_VAR_project_id \
      --member="serviceAccount:neg-keywords-cleaner@$TF_VAR_project_id.iam.gserviceaccount.com" \
      --role="roles/run.invoker"
    gcloud projects add-iam-policy-binding $TF_VAR_project_id \
      --member="serviceAccount:neg-keywords-cleaner@$TF_VAR_project_id.iam.gserviceaccount.com" \
      --role="roles/storage.admin"
    ```
7. Create an Artifact Registry repository.
    ```
    gcloud artifacts repositories create negatives \
      --repository-format=docker \
      --location=us-central1
    ```
8. Replace the ... with the Client ID and Client Secret obtained in Step 2
   ```
   echo 'google_oauth_client_id = "..."' > terraform/secrets.tfvars
   echo 'google_oauth_client_secret = "..."' >> terraform/secrets.tfvars
   echo 'mcc_id = "XXXXXXXXXX"' >> terraform/secrets.tfvars
   echo 'google_ads_api_token = "..."' >> terraform/secrets.tfvars
   ```
   > You can edit the ```terraform/main.tf``` file under the
   google_iam_policy.noauth section, to restrict access to specific users (the
   app
   will be visible to all users with the url per default).

9. Build and deploy container
   ```
   docker build -t us-central1-docker.pkg.dev/$TF_VAR_project_id/negatives/negatives:v1 . --no-cache
   docker push us-central1-docker.pkg.dev/$TF_VAR_project_id/negatives/negatives:v1
   ```
   > If it doesn't deploy, you might have to run `gcloud auth configure-docker us-central1-docker.pkg.dev`
   and
   try again.

10. Deploy the solution
    ```
    cd terraform/
    terraform init -upgrade
    terraform apply -var-file secrets.tfvars
    ```
    > Take note of the **App URL**. *Example: https://myapp.run.app*

11. Go back to Credentials > OAuth 2.0 Client ID and select your Client ID.
    * Add the App URL into the list of **Authorized Redirect URL**.

## Uninstall

You can remove all GCP resources with one command:

    terraform destroy

## Run the app on a local machine

1. **Configure environment variables**: Create a `.env` file in the root directory with the following content (replace with your credentials):
   ```env
   OAUTH_CLIENT_ID=your_client_id
   OAUTH_CLIENT_SECRET=your_client_secret
   OAUTH_REDIRECT_URI=http://localhost:8080
   MCC_ID=your_mcc_id
   GOOGLE_ADS_API_TOKEN=your_dev_token
   GOOGLE_API_KEY=your_gemini_api_key
   ```

2. **Install and run**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt --extra-index-url https://pypi.org/simple
   streamlit run src/AI_student.py
   ```

*Note: This project requires Python 3.11. If you need to manage Python versions, consider using `pyenv`.*

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
