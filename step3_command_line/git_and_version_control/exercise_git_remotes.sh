## 1. Git remotes ##

/home/dq$ git clone /dataquest/user/git/chatbot

## 2. Making changes to cloned repositories ##

/home/dq/chatbot$ git status

## 3. The master branch ##

/home/dq/chatbot$ git branch

## 4. Pushing changes to the remote ##

/home/dq/chatbot$ git push origin master

## 5. Viewing individual commits ##

/home/dq/chatbot$ git show 3d0af2f8efdab441138da8d16b9d2a7ceaf238c9

## 6. Commits and the working directory ##

/home/dq/chatbot$ git diff

## 7. Switching to a specific commit ##

/home/dq/chatbot$ git reset --hard ef4481222ce2e533f8f576bf6f1c346174ae67d1

## 8. Pulling from a remote repo ##

/home/dq/chatbot$ git pull origin master

## 9. Referring to the most recent commit ##

/home/dq/chatbot$ git reset --hard HEAD~1