# German Bank Customer Segmentation
A clustering web app which leverages KMeans to organize german bank customers and using FastAPI for robust API functionality and Docker for containerization.

In this project, I have organized german bank customers in clusters based on three main features: Credit_amount, Age and Duration of the credit. I decided to robust 
this project's functionality by connecting it to a web framework for building APIs such as FastAPI. In the end to guarantee that the app runs across different environments
i dockerized it. 

The input passes as json and i have used Postman for the testing. Feel free to test the app, and also add other functionalities.

`git clone  https://github.com/GSana2812/German_Bank_Customer_Segmentation.git`

To access the app:  `docker compose up --build` 

Libraries used: ipykernel, numpy, pandas, matplotlib, scikit-learn, fastapi, pydantic, uvicorn, seaborn.
