Unit tests are very slow to run. This is causing development to slow down each time we need to run all of them.

- It seems to be downloading the models many times. That is unneccessary. I don't want to download the file each time as part of the standard unit tests. This would be an integration test.
- It is probably indexing more files than are necessary for the tests.
- Identify slow test suites and remedy
- Don't let coverage go down
- Keep the same functional test footprint
