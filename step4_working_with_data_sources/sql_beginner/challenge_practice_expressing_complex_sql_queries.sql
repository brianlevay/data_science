## 2. Select and Limit ##

SELECT College_jobs, Median, Unemployment_rate
FROM recent_grads
LIMIT 20;

## 3. Where ##

SELECT Major
FROM recent_grads
WHERE Major_category == "Arts"
LIMIT 5;

## 4. Operators ##

SELECT Major, Total, Median, Unemployment_rate
FROM recent_grads
WHERE (Major_category != "Engineering") AND (Median <= 50000 OR Unemployment_rate > 0.065);

## 5. Ordering ##

SELECT Major
FROM recent_grads
WHERE Major_category != "Engineering"
ORDER BY Major DESC
LIMIT 20;