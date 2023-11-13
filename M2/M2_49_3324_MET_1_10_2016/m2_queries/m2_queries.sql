/* 1. All trip info(location,tip amount,etc) for the 20 highest trip distances. */
SELECT *
FROM green_taxi_10_2016 inner join 
ORDER BY distance DESC
LIMIT 20;

Select  GT.pickup_datetime, GT.dropoff_datetime, 
 		locPu."Original value" As "Pu Location", locDo."Original value" As "Do Location",
		taxi."Passenger Count", taxi."Trip Distance", taxi."Fare Amount", taxi."Extra",
		taxi."Mta Tax", taxi."Tip Amount", taxi."Tolls Amount", taxi."Ehail Fee", taxi."Improvement Surcharge",
		taxi."Total Amount", look2."Original value" As "Payment Type", look3."Original value" As "Trip Type",
		taxi."Congestion Surcharge", taxi."Weekday", taxi."Duration", taxi."Velocity", taxi."Negative Money",
		taxi."Week Number", taxi."Date Range", taxi."Vendor Creative Mobile Technologies, Llc",
		taxi."Vendor Verifone Inc.", taxi."Store And Fwd Flag N", taxi."Store And Fwd Flag Y", taxi."Is Weekend Trip",
		look4."Original value" As "Trip Period", 
		taxi."Pu Location Latitude", taxi."Pu Location Longtitude",
		taxi."Do Location Latitude", taxi."Do Location Longtitude"
		
       	From green_taxi_10_2016 As GT

	   Order by taxi.trip_distance Desc
	   Limit 20


/* 2. What is the average fare amount per payment type. */
SELECT 
    CASE
        WHEN "encoded_payment_type_Credit card" = 1 THEN 'Credit Card'
        WHEN "payment_type_Cash" = 1 THEN 'Cash'
        WHEN "payment_type_Dispute" = 1 THEN 'Dispute'
        WHEN "payment_type_No charge" = 1 THEN 'No Charge'
        ELSE 'Unknown'
    END AS payment_type,
    AVG(fare_amount) AS average_fare
FROM 
    green_taxi_10_2016
GROUP BY 
    "encoded_payment_type_Credit card", "payment_type_Cash","payment_type_Dispute","payment_type_No charge";


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




/* 6. On average, which trip type travels longer distances. */



/* 7. between 4pm and 6pm what is the average fare amount. */