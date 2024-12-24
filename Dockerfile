FROM python:3.14.0a3 
WORKDIR /bot
ADD . /bot
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]