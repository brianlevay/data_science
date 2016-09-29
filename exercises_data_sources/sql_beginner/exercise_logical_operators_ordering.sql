## 2. And operator ##

SELECT Major,ShareWomen,Employed FROM recent_grads 
WHERE ShareWomen>0.5 AND Employed>10000
LIMIT 10;

## 3. Or operator ##

SELECT Major, Median, Unemployed
FROM recent_grads
WHERE Median >= 10000 OR Unemployed < 1000
LIMIT 20;

## 4. Grouping operators ##

select Major, Major_category, ShareWomen, Unemployment_rate
from recent_grads
where (Major_category = 'Engineering') and (ShareWomen > 0.5 or Unemployment_rate < 0.051);

## 5. Practice grouping operators ##

SELECT Major, Major_category, Employed, Unemployment_rate
FROM recent_grads
WHERE (Major_category == "Business" OR Major_category == "Arts" OR Major_category == "Health") 
AND (Employed > 20000 OR Unemployment_rate < 0.051);

## 6. Order by ##

SELECT Major
FROM recent_grads
ORDER BY Major DESC
LIMIT 10;

## 7. Order using multiple columns ##

SELECT Major_category, Median, Major
FROM recent_grads
ORDER BY Major ASC, Median DESC
LIMIT 20;