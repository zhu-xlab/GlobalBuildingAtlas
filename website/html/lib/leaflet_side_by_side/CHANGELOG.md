# Change Log

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased

- ADDED: Allows Leaflet version 0.7.7 through 1.x

## [v2.0.0] - 2015-12-08

- ADDED: Add `setLeftLayers()` and `setRightLayers()` methods
- ADDED: `options.padding`
- ADDED: `getPosition()` returns the x coordinate (relative to the map container) of the divider
- FIXED: **[BREAKING]** Export factory function on `L.control` not `L.Control`
- FIXED: Slider drag was not working on touch devices

## [v1.1.1] - 2015-12-03

- FIXED: fix package.json settings for npm distribution

## [v1.1.0] - 2015-12-03

- ADDED: Events
- FIXED: Fix initial divider position in Firefox, should start in middle of map

## v1.0.2 - 2015-12-02

Initial release

[Unreleased]: https://github.com/digidem/leaflet-side-by-side/compare/v2.0.0...HEAD
[Unreleased]: https://github.com/digidem/leaflet-side-by-side/compare/v1.1.1...v2.0.0
[v1.1.1]: https://github.com/digidem/leaflet-side-by-side/compare/v1.1.0...v1.1.1
[v1.1.0]: https://github.com/digidem/leaflet-side-by-side/compare/v1.0.2...v1.1.0
