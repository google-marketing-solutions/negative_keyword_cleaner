FROM python:3.11.3

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
RUN find /usr/local/lib/python3.11/site-packages/streamlit -type f \( -iname \*.py -o -iname \*.js \) -print0 | xargs -0 sed -i 's/healthz/health-check/g'
COPY src/ ./src/
RUN touch ./app_config.yaml

EXPOSE 8080
CMD streamlit run --server.port 8080 --browser.serverAddress 0.0.0.0 --server.enableCORS False --server.enableXsrfProtection False src/AI_student.py
