FROM python:3.6-stretch

RUN useradd -ms /bin/bash publishme

WORKDIR /home/publishme

COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN pip install --upgrade gensim
RUN pip install gunicorn

RUN echo "#/bin/sh" >> ./boot.sh
RUN echo "flask run" >> ./boot.sh

COPY app app
RUN chmod +x boot.sh

ENV FLASK_APP app.py

RUN chown -R publishme:publishme ./
USER publishme

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]