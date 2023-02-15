# fakeNews

To run the app you will need to: 
    MongoDB:
        docker-compose.yaml up - if you have docker
        if you have MongoDB installed alter these 2 lines
            on app.py line 8 alter the "rootuser:rootpass" to your mongoDB credentials
    submit2mongo:
        run this script to upload the labes of the test.csv

The option to upload CSV only works with the above csv from the 'data' folder:
    -test.csv
    -test1.csv
    -test2.csv