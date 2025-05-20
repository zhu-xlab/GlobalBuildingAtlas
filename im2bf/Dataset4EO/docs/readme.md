## Edit docs
Notes about editing the docs.
### Header Logo

We suggest putting all the html resources to `docs/_static`. First put the logo
to `docs/_static/images/logo.png`, then write the following snippet to 
`docs/_static/css/readthedocs.css`:
```css
.header-logo {
    background-image: url("../images/logo.png");
    background-size: 110px 40px;
    height: 40px;
    width: 110px;
}
```
Here, you are recommended to fix the height to `40px` and scale the width according to the logo's aspect ratio.
The latest thing to do is to tell Sphinx the location of these resources by adding the following lines to `docs/conf.py`:
```python
html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']
```

### Header Customization
This theme variant also allows users to customize the header, such as the logo url and the navigation menu, in a pythonic way. They are all configurable options in `html_theme_options` in `docs/conf.py`.

Here is an example config covering all available options:
```python
html_theme_options = {
    # The target url that the logo directs to. Unset to do nothing
    'logo_url': 'https://mmocr.readthedocs.io/en/latest/',
    # "menu" is a list of dictionaries where you can specify the content and the 
    # behavior of each item in the menu. Each item can either be a link or a
    # dropdown menu containing a list of links.
    'menu': [
        # A link
        {
            'name': 'GitHub',
            'url': 'https://github.com/open-mmlab/'
        }, 
        # A dropdown menu
        {
            'name': 'Projects',
            'children': [
                # A vanilla dropdown item
                {
                    'name': 'MMCV',
                    'url': 'https://github.com/open-mmlab/mmcv',
                },
                # A dropdown item with a description
                {
                    'name': 'MMDetection',
                    'url': 'https://github.com/open-mmlab/mmdetection',
                    'description': 'Object detection toolbox and benchmark'
                },
            ], 
            # Optional, determining whether this dropdown menu will always be
            # highlighted. 
            'active': True,
        },
    ],
    # For shared menu: If your project is a part of OpenMMLab's project and 
    # you would like to append Docs and OpenMMLab section to the right
    # of the menu, you can specify menu_lang to choose the language of
    # shared contents. Available options are 'en' and 'cn'. Any other
    # strings will fall back to 'en'.
    'menu_lang':
    'en',
}
```

### Shared menu

You have to edit `pytorch_sphinx_theme` library. Go to `pytorch_sphinx_theme\pytorch_sphinx_theme\theme_variables.jinja`.

### Side bar

Edit `index.rst`.


## Docs TODOs

- [ ] Edit the contents of the page.
- [ ] Edit the tutorial.
- [ ] Put on `readthedocs.io` and link to EarthNets web page.