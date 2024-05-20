FROM python:3.10
WORKDIR /code
ENV PORT 8000
EXPOSE 8000
COPY . /code
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
    
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]