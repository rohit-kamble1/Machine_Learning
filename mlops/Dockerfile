FROM python:3.8-slim-buster
COPY . /app
WORKDIR /app
RUN apt update -y && apt install awscli -y
RUN pip install -r requirement.txt
EXPOSE 8501
CMD streamlit run app.py