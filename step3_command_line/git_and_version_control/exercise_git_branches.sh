## 1. Git branches ##

/home/dq/chatbot$ git checkout -b more-speech

## 2. Switching branches ##

/home/dq/chatbot$ git commit -m "Added more output"

## 3. Pushing a branch to a remote ##

/home/dq/chatbot$ git push origin more-speech

## 4. Merging branches ##

/home/dq/chatbot$ git push origin master

## 5. Deleting branches ##

/home/dq/chatbot$ git branch -d more-speech

## 6. Checking out branches from the remote ##

/home/dq/chatbot2$ git push origin happy-bot

## 7. Finding differences across branches ##

/home/dq/chatbot2$ git diff master happy-bot

## 8. Branch naming conventions ##

/home/dq/chatbot$ git push origin feature/random-messages

## 9. Branch history ##

/home/dq/chatbot$ git checkout -b feature/spam-messages