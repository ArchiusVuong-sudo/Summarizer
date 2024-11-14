# Document Extraction
- Download the files and put them into the `docs` folder.
- Execute the script using the below commands.
- The output will be stored into an `xlsx` file.

```
docker build -t document-extraction .
docker run -it --rm -v "$(pwd)/output:/app/output" document-extraction
```
