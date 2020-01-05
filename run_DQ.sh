#!/usr/bin/env bash

export MAVEN_OPTS="-Xmx1024m -XX:MaxPermSize=128m"
mvn exec:java -Dexec.mainClass="Main" -Dexec.args="CQL"
#cd target/classes
#java Main
