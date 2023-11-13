/* 1. All trip info(location,tip amount,etc) for the 20 highest trip distances. */
SELECT *
FROM green_taxi_10_2016
ORDER BY distance DESC
LIMIT 20;


/* 2. What is the average fare amount per payment type. */
SELECT payment_type, AVG(fare_amount) AS average_fare
FROM green_taxi_10_2016
GROUP BY payment_type;


/* 3. On average, which city tips the most.*/


SELECT pu_location AS city, AVG(tip_amount) AS average_tip
FROM green_taxi_10_2016
GROUP BY pu_location
ORDER BY average_tip DESC
LIMIT 1;


/* 4. On average, which city tips the least.*/

SELECT pu_location AS city, AVG(tip_amount) AS average_tip
FROM green_taxi_10_2016
GROUP BY pu_location
ORDER BY average_tip ASC
LIMIT 1;

/* 5. What is the most frequent destination on the weekend.*/




