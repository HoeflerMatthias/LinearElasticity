FROM firedrakeproject/firedrake-vanilla-default:latest

COPY requirements.txt /tmp/requirements.txt

# Use the venv python/pip to install your extra packages into the same env
RUN pip install -r /tmp/requirements.txt

WORKDIR /app