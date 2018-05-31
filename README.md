# jcuRandomWalk
jCuda approach to random walk segmentation

## Building the project

```sh
mvn package -P generate-exe
```

### Releasing

```sh
cd jcuRandomWalk 
mvn clean release:prepare -DpushChanges=false
mvn release:perform
mvn site
github_changelog_generator -t $(cat ~/.git-credentials | grep -Po '(?<=CellDynamics:).*(?=@)')
git add CHANGELOG.md
git commit -m "Added changelog"
git push 

# upload binary
git checkout $TAG
mvn clean package -P generate-exe
gzip -9 target/jcuRandomWalk-x.x.x.jar
# upload to GH
```

## Running

```sh
java -jar target/jcuRandomWalk-0.0.1-SNAPSHOT.jar -h
```
