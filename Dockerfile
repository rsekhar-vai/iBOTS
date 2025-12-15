FROM python:3.14.2 
WORKDIR /bot
ADD . /bot
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]