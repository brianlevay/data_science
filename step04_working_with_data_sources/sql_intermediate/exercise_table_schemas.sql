## 2. Adding columns ##

ALTER TABLE facts 
ADD leader TEXT;

## 6. Creating a table with relations ##

CREATE TABLE factbook.states(
    id INTEGER PRIMARY KEY,
    name TEXT,
    area INTEGER,
    country INTEGER REFERENCES facts(id)
);

## 7. Querying across foreign keys ##

SELECT * 
FROM landmarks 
INNER JOIN facts 
ON landmarks.country == facts.id;

## 8. Types of joins ##

SELECT * 
FROM landmarks 
LEFT OUTER JOIN facts 
ON landmarks.country == facts.id;