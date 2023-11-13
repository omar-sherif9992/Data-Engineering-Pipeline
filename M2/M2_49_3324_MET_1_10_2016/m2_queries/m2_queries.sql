/* 1. All trip info(location,tip amount,etc) for the 20 highest trip distances. */
-- All trip info(location, tip amount, etc) for the 20 highest trip distances.
-- I am joining the labeled encoded columns to be interpretable , 
SELECT 
    L1.original AS pickup_location,
    L2.original AS dropoff_location,
    GT.*
FROM 
    green_taxi_10_2016 AS GT
INNER JOIN 
    lookup_green_taxi_10_2016 AS L1 
ON   
    GT.encoded_pu_location::text = L1.encoded 
INNER JOIN 
    lookup_green_taxi_10_2016 AS L2
ON
    GT.encoded_do_location::text = L2.encoded

WHERE 
    L1.column_name = 'encoded_location' AND 
    L2.column_name = 'encoded_location'   
ORDER BY 
    GT.trip_distance DESC
LIMIT 20;


/* 2. What is the average fare amount per payment type. */

SELECT 
    CASE
        WHEN "encoded_payment_type_Credit card" THEN 'Credit Card'
        WHEN "encoded_payment_type_Cash" THEN 'Cash'
        WHEN "encoded_payment_type_Dispute" THEN 'Dispute'
        WHEN "encoded_payment_type_No charge" THEN 'No Charge'
        ELSE 'Unknown'
    END AS payment_type,
    AVG(fare_amount) AS average_fare
FROM 
    green_taxi_10_2016
GROUP BY 
    "encoded_payment_type_Credit card", "encoded_payment_type_Cash", "encoded_payment_type_Dispute", "encoded_payment_type_No charge";


/* 3. On average, which city tips the most.*/
SELECT 
    SPLIT_PART(L.original, ',', 1) AS city,
    AVG(GT.tip_amount) AS average_tip
FROM 
    green_taxi_10_2016 AS GT
INNER JOIN 
    lookup_green_taxi_10_2016 AS L 
ON   
    GT.encoded_pu_location::text = L.encoded OR GT.encoded_do_location::text = L.encoded
WHERE 
    L.column_name = 'encoded_location'
GROUP BY 
    city
ORDER BY 
    average_tip DESC
LIMIT 1;


/* 4. On average, which city tips the least.*/


SELECT 
    SPLIT_PART(L.original, ',', 1) AS city,
    AVG(GT.tip_amount) AS average_tip
FROM 
    green_taxi_10_2016 AS GT
INNER JOIN 
    lookup_green_taxi_10_2016 AS L 
ON   
    GT.encoded_pu_location::text = L.encoded OR GT.encoded_do_location::text = L.encoded
WHERE 
    L.column_name = 'encoded_location'
GROUP BY 
    city
ORDER BY 
    average_tip ASC
LIMIT 1;


/* 5. What is the most frequent destination on the weekend.*/
SELECT 
    L.original as 'destination',
    COUNT(GT.*) AS destination_count
FROM 
    green_taxi_10_2016 AS GT
INNER JOIN 
    lookup_green_taxi_10_2016 AS L 
ON   
   GT.encoded_do_location::text = L.encoded

WHERE 
    EXTRACT(DAY FROM TO_TIMESTAMP(GT.lpep_pickup_datetime,'YYYY-MM-DD HH24:MI:SS')) IN (0, 6) AND -- 0 is Sunday, 6 is Saturday
	L.column_name = 'encoded_location'

GROUP BY 
    L.original
ORDER BY 
    destination_count DESC
LIMIT 1;




/* 6. On average, which trip type travels longer distances. */

SELECT 
    CASE
        WHEN "trip_type_1" = 1 THEN 'Trip Type 1'
        WHEN "trip_type_2" = 1 THEN 'Trip Type 2'
        WHEN "trip_type_3" = 1 THEN 'Trip Type 3'
        -- Add more WHEN clauses for other trip types if necessary
        ELSE 'Unknown'
    END AS trip_type,
    AVG(trip_distance) AS average_distance
FROM 
    green_taxi_10_2016
GROUP BY 
    "trip_type_1", "trip_type_2", "trip_type_3";
order by average_distance DESC

/* 7. between 4pm and 6pm what is the average fare amount. */