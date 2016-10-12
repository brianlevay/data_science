## 2. Authenticating with the API ##

headers = {"Authorization": "bearer 13426216-4U1ckno9J5AiK72VRbpEeBaMSKk", "User-Agent": "Dataquest/1.0"}
params = {"t": "day"}

response = requests.get("https://oauth.reddit.com/r/python/top", headers=headers, params=params)

python_top = response.json()
print(python_top)

## 3. Getting the most upvoted article ##

python_top_articles = python_top["data"]["children"]

most_ups = python_top_articles[0]["data"]["ups"]
most_upvoted = python_top_articles[0]["data"]["id"]

for article in python_top_articles:
    if article["data"]["ups"] > most_ups:
        most_ups = article["data"]["ups"]
        most_upvoted = article["data"]["id"]

print("Most ups: {0}, Article ID: {1}".format(most_ups, most_upvoted))

## 4. Getting article comments ##

base_url = "https://oauth.reddit.com/r/python/comments"
full_url = base_url + "/" + most_upvoted
response = requests.get(full_url, headers=headers)
comments = response.json()
print(comments)

## 5. Getting the most upvoted comment ##

top_lvl = comments[1]["data"]["children"]

most_ups = top_lvl[0]["data"]["ups"]
most_upvoted_comment = top_lvl[0]["data"]["id"]

for comment in top_lvl:
    if comment["data"]["ups"] > most_ups:
        most_ups = comment["data"]["ups"]
        most_upvoted_comment = comment["data"]["id"]

print("Most ups: {0}, Comment ID: {1}".format(most_ups, most_upvoted_comment))

## 6. Upvoting a comment ##

payload = {"dir": 1, "id": most_upvoted_comment}
response = requests.post("https://oauth.reddit.com/api/vote", headers=headers, json=payload)
status = response.status_code
print(status)