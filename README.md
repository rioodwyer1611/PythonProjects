# PythonProjects
A collection of data science projects completed during my journey to master Python and Data Science.

## Site Navigation

This site features a sidebar navigation that allows easy access to different pages. The navigation is fixed on desktop screens and slides in from the left on mobile devices.

### Current Pages
- **Home** (`index.html`) - Main landing page with project overview
- **Projects** (`projects.html`) - Dedicated page showcasing all projects
- **About Me** (`about.html`) - Personal information and skills

### Adding a New Page

1. **Create the HTML file**: Copy an existing page (e.g., `about.html`) and rename it to your new page name (e.g., `contact.html`).

2. **Update the page content**: Modify the `<title>`, hero section, and main content to match your new page.

3. **Update the active navigation link**: In your new page, find the `nav-link` for your page and add the `active` class, while removing it from other links.

4. **Add the navigation link to all pages**: Add a new `<li>` item to the `<ul class="nav-links">` section in every HTML file:

```html
<li><a href="your-page.html" class="nav-link">
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <!-- Add your icon SVG path here -->
    </svg>
    Your Page Name
</a></li>
```

5. **Choose an icon**: You can find free SVG icons at [Feather Icons](https://feathericons.com/) or similar icon libraries.

## About Me
#### Name: Rio O'Dwyer

![Profile Photo](images/example-profile.svg)

A passionate data science enthusiast on a journey to master Python and data analysis techniques.

- Location: Brisbane, Australia
- Interests: Data Science, Machine Learning, Python Development
- Currently Studying: Bachelor of Computer Science/Master of Data Science (University of Queensland)
- Other Studies: IBM Data Science Professional Certificate
- GitHub: rioodwyer1611 (https://github.com/rioodwyer1611)

## [Stock Grapher:](https://github.com/rioodwyer1611/PythonProjects/tree/main/Stock_Grapher)
#### A program which retrieves current share values of Tesla and GameStop stocks and visualizes their stock data alongside revenue output through interactive graphs.

![Stock Grapher Preview](images/stock-grapher-placeholder.jpg)

Created to learn how to use:
  - Pandas
  - BeautifulSoup
  - Requests
  - YFinance

Completed as part of an IBM Python Project for Data Science Course.


## [Optimal Coffee Shop Location Finder:](https://github.com/rioodwyer1611/PythonProjects/tree/main/Optimal_Coffee_Shop_Location_Finder)
#### An application that displays all 81 libraries in Chicago and uses optimization algorithms to identify the best locations for opening coffee shops near these libraries.

![Coffee Shop Finder Preview](images/coffee-shop-placeholder.jpg)

Created to learn:
  - Requests
  - Folium
  - GeoPy
  - Webbrowser
  - OS
  - DocPlex & CPlex (Not used in final model but available to view in code)
  - OrTools

Completed as a part of an IBM What is Data Science? Course.

## [Weather Data Analyser:](https://github.com/rioodwyer1611/PythonProjects/tree/main/Weather_Data_Analyser)
#### A data cleaning, analysis and graphing project that demonstrates the basics of data preprocessing and visualization techniques using weather datasets.

![Weather Data Analyser Preview](images/weather-data-placeholder.jpg)

Created to learn:
  - Pandas
  - MatPlotLib
  - SeaBorn
  - NumPy

Taken from IBM Tools for Data Science Course, altered to fit current pandas formats.

## [Basic Jupyter Notebook:](https://github.com/rioodwyer1611/PythonProjects/tree/main/Weather_Data_Analyser)
#### A super basic Jupyter notebook created to learn how to properly use the development tool.

![Weather Data Analyser Preview](images/weather-data-placeholder.jpg)

Created to learn:
  - Jupyter

Completed as a part of IBM Tools for Data Science Course.
