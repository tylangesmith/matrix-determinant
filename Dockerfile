FROM python:3.8
RUN mkdir determinant
WORKDIR determinant
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["python","-m","pytest","."]