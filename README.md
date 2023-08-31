# Labels

Code for converting JSON data containing vector graphics and text data exported from Adobe Illustrator into a transform compatible with the [sheet-to-triples](https://github.com/VisualMeaning/sheet-to-triples) tool. This data is ingested as part of an RDF graph model and used to display labels on top of maps displayed as part of the Shared Meaning Platform.

## Setup

To install, first ensure that Python version 3.10 or greater is installed. This code currently works with no additional packages required beyond base Python packages.

A stub install command is provided:

    make install

This currently does nothing, but if additional packages are required in the future then this command will ensure that they are installed.

## Run

    python3 labels.py IllustratorExport.json --output labels_transform.py

This will take the export file `IllustratorExport.json` as an input and use it to produce the labels transform `labels_transform.py`.

### Common arguments

* `-d`, `--lens-detail`
  Flags that the Illustrator export file uses lenses and detail levels rather than base Illustrator layers. Should be used with all export files generated from our semantic tooling package.

* `-q`, `--quiet`
  By default the labels script will print the labels tree to console as it is processed. This argument suppresses that output.

* `--output <output file name>`
  By default the labels script will print its transform data to stdout once it has been processed. This output can be piped to a file as desired, however this functionality doesn't work well on some OSes. If `--output` is passed, Python file write functionality will be used to write the transform data to the nominated output file instead of printing to stdout.

* `--export-tabular`
  Create a TSV-format file containing the labels tree data for subsequent inspection.

There are many other arguments with specific uses, run `python3 labels.py --help` for the full list.
