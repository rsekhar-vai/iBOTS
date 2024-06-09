FROM python:3.12.3 
WORKDIR /bot
ADD . /bot
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]