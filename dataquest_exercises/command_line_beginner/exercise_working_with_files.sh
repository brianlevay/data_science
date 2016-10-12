## 1. Making a file ##

/home/dq$ touch test.txt

## 2. Standard streams ##

/home/dq$ echo "All bears should juggle"

## 3. Redirection ##

/home/dq$ echo "All bears should juggle" > test.txt

## 4. Editing a file ##

/home/dq$ nano test.txt

## 5. File permissions ##

/home/dq$ ls -l

## 6. Octal notation ##

/home/dq$ stat test.txt

## 7. Modifying file permissions ##

/home/dq$ chmod 0760 test.txt

## 8. Moving files ##

/home/dq$ mv test.txt test

## 9. Copying files ##

/home/dq$ cp test/test.txt test/test2.txt

## 10. File extensions ##

/home/dq$ mv test/test.txt test/test_no_extension

## 11. Deleting a file ##

/home/dq$ rm test/test2.txt

## 12. The root user ##

/home/dq$ sudo mkdir test3