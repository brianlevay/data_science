## 4. Querying a normalized database ##

query = "\
SELECT ceremonies.year, nominations.movie \
FROM nominations \
INNER JOIN ceremonies \
ON ceremonies.id = nominations.ceremony_id \
WHERE nominations.nominee == 'Natalie Portman';"

c = conn.cursor()
c.execute(query)
portman_movies = c.fetchall()
print(portman_movies)

## 6. Join table ##

c = conn.cursor()

query1 = "SELECT * FROM movies_actors LIMIT 5;"
c.execute(query1)
five_join_table = c.fetchall()

query2 = "SELECT * FROM actors LIMIT 5;"
c.execute(query2)
five_actors = c.fetchall()

query3 = "SELECT * FROM movies LIMIT 5;"
c.execute(query3)
five_movies = c.fetchall()

print(five_join_table)
print(five_actors)
print(five_movies)

## 7. Querying a many-to-many relation ##

c = conn.cursor()

query = "\
SELECT actors.actor, movies.movie FROM movies \
INNER JOIN movies_actors ON movies.id == movies_actors.movie_id \
INNER JOIN actors ON movies_actors.actor_id == actors.id \
WHERE movies.movie == 'The King''s Speech';"

c.execute(query)
kings_actors = c.fetchall()
print(kings_actors)

## 8. Practice: querying a many-to-many relation ##

c = conn.cursor()

query = "\
SELECT movies.movie, actors.actor FROM movies \
INNER JOIN movies_actors ON movies.id == movies_actors.movie_id \
INNER JOIN actors ON movies_actors.actor_id == actors.id \
WHERE actors.actor == 'Natalie Portman';"

c.execute(query)
portman_joins = c.fetchall()
print(portman_joins)