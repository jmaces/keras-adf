# Pull Request Check List

This is a reminder about the most common mistakes. Please tick all _appropriate_ boxes. Also please read our [contribution guide](https://github.com/jmaces/keras-adf/blob/master/.github/CONTRIBUTING.rst) at least once, it will save you unnecessary review cycles!

If an item does not apply to your pull request, **check it anyway** to make it apparent that there is nothing to worry about.

- [ ] Added **tests** for changed code.
- [ ] Updated **documentation** for changed code.
  - [ ] New functions/classes have to be added to the API reference.
  - [ ] Changed/added classes/methods/functions have appropriate `versionadded`, `versionchanged`, or `deprecated` [directives](http://www.sphinx-doc.org/en/stable/markup/para.html#directive-versionadded).
- [ ] Documentation in `.rst` files is written using [semantic newlines](https://rhodesmill.org/brandon/2012/one-sentence-per-line/).
- [ ] Changes (and possible deprecations) have news fragments in [`changelog.d`](https://github.com/jmaces/keras-adf/blob/master/changelog.d).

If you have _any_ questions to _any_ of the points above, just **submit and ask**! This checklist is here to _help_ you, not to deter you from contributing!
