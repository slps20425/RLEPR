# Use the official Python 3.6 image as the base image
FROM python:3.6

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Set the timezone
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy the entire folder structure into the container
COPY . .

# Expose port 8050 for Dash app
EXPOSE 8050

# Run the entrypoint.sh script when the container launches
CMD ["/bin/bash", "entrypoint.sh"]

