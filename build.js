const fs = require('fs');
const path = require('path');

// Placeholder images from Unsplash (coding-related)
const PLACEHOLDER_IMAGES = {
    profile: 'https://images.unsplash.com/photo-1517694712202-14dd9538aa97?w=400&h=400&fit=crop',
    stock: 'https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=600&h=400&fit=crop',
    coffee: 'https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=600&h=400&fit=crop',
    weather: 'https://images.unsplash.com/photo-1504608524841-42fe6f032b4b?w=600&h=400&fit=crop',
    data: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=600&h=400&fit=crop',
    code: 'https://images.unsplash.com/photo-1555066931-4365d14bab8c?w=600&h=400&fit=crop',
    map: 'https://images.unsplash.com/photo-1524661135-423995f22d0b?w=600&h=400&fit=crop'
};

// Read the README file
const readmePath = path.join(__dirname, 'README.md');
const readmeContent = fs.readFileSync(readmePath, 'utf8');

// Parse the README to extract personal info and projects
function parseReadme(content) {
    const lines = content.split('\n');
    const result = {
        title: '',
        description: '',
        personalInfo: null,
        projects: []
    };

    let currentSection = null;
    let currentProject = null;
    let inPersonalInfo = false;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        // Main title (# PythonProjects)
        if (line.startsWith('# ') && !result.title) {
            result.title = line.substring(2).trim();
            continue;
        }

        // Description (first paragraph after title)
        if (!result.description && result.title && line.trim() && !line.startsWith('#') && !line.startsWith('-')) {
            result.description = line.trim();
            continue;
        }

        // Personal Info Section (## About Me or ## Personal Info)
        if (line.match(/^## \[?(About Me|Personal Info|About)\]?/i)) {
            inPersonalInfo = true;
            result.personalInfo = {
                name: '',
                bio: '',
                details: [],
                image: ''
            };
            continue;
        }

        // Project section (## [Project Name:](url))
        const projectMatch = line.match(/^## \[([^\]]+)\]\(([^)]+)\)/);
        if (projectMatch) {
            inPersonalInfo = false;
            if (currentProject) {
                result.projects.push(currentProject);
            }
            currentProject = {
                name: projectMatch[1].replace(/:$/, ''),
                url: projectMatch[2],
                description: '',
                techStack: [],
                course: '',
                featured: false,
                hasMap: false,
                image: ''
            };
            continue;
        }

        // Handle personal info content
        if (inPersonalInfo && result.personalInfo) {
            // Name line (#### Name: Rio O'Dwyer)
            const nameMatch = line.match(/^####?\s*Name:\s*(.+)/i);
            if (nameMatch) {
                result.personalInfo.name = nameMatch[1].trim();
                continue;
            }

            // Image reference
            const imageMatch = line.match(/!\[.*\]\(([^)]+)\)/);
            if (imageMatch) {
                result.personalInfo.image = imageMatch[1];
                continue;
            }

            // Bio line - plain text that isn't a list or header
            if (line.trim() && !line.startsWith('#') && !line.startsWith('-') && !line.startsWith('!') && !result.personalInfo.bio) {
                result.personalInfo.bio = line.trim();
                continue;
            }

            // Detail items (- key: value)
            const detailMatch = line.match(/^\s*-\s*(.+):\s*(.+)/);
            if (detailMatch) {
                result.personalInfo.details.push({
                    key: detailMatch[1].trim(),
                    value: detailMatch[2].trim()
                });
                continue;
            }

            // Simple list items for bio
            const simpleDetailMatch = line.match(/^\s*-\s*(.+)/);
            if (simpleDetailMatch && !line.includes(':')) {
                result.personalInfo.details.push({
                    key: '',
                    value: simpleDetailMatch[1].trim()
                });
            }
            continue;
        }

        // Handle project content
        if (currentProject) {
            // Description line (#### followed by description)
            if (line.startsWith('#### ')) {
                currentProject.description = line.substring(5).trim();
                continue;
            }

            // Tech stack items
            const techMatch = line.match(/^\s*-\s*(.+)/);
            if (techMatch && !line.toLowerCase().includes('completed')) {
                const tech = techMatch[1].trim();
                // Skip parenthetical notes
                if (!tech.startsWith('(')) {
                    currentProject.techStack.push(tech.split('(')[0].trim());
                }
                continue;
            }

            // Course info
            if (line.toLowerCase().includes('completed as')) {
                currentProject.course = line.trim();
                continue;
            }

            // Check for featured or map indicators
            if (line.toLowerCase().includes('featured')) {
                currentProject.featured = true;
            }
            if (line.toLowerCase().includes('folium') || line.toLowerCase().includes('map')) {
                currentProject.hasMap = true;
            }

            // Image reference
            const imageMatch = line.match(/!\[.*\]\(([^)]+)\)/);
            if (imageMatch) {
                currentProject.image = imageMatch[1];
            }
        }
    }

    // Don't forget the last project
    if (currentProject) {
        result.projects.push(currentProject);
    }

    // Mark Coffee Shop project as featured and having map
    result.projects.forEach(project => {
        if (project.name.toLowerCase().includes('coffee') || project.name.toLowerCase().includes('location')) {
            project.featured = true;
            project.hasMap = true;
        }
        // Check tech stack for folium
        if (project.techStack.some(t => t.toLowerCase().includes('folium'))) {
            project.hasMap = true;
        }
    });

    return result;
}

// Generate project card HTML
function generateProjectCard(project, index) {
    const icons = {
        stock: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>`,
        map: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
            <circle cx="12" cy="10" r="3"></circle>
        </svg>`,
        weather: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"></path>
        </svg>`,
        data: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 3v18h18"></path>
            <path d="m19 9-5 5-4-4-3 3"></path>
        </svg>`,
        code: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="16 18 22 12 16 6"></polyline>
            <polyline points="8 6 2 12 8 18"></polyline>
        </svg>`
    };

    // Choose icon and placeholder image based on project name
    let icon = icons.code;
    let placeholderImage = PLACEHOLDER_IMAGES.code;
    const nameLower = project.name.toLowerCase();

    if (nameLower.includes('stock') || nameLower.includes('graph')) {
        icon = icons.stock;
        placeholderImage = PLACEHOLDER_IMAGES.stock;
    } else if (nameLower.includes('coffee') || nameLower.includes('location') || nameLower.includes('map')) {
        icon = icons.map;
        placeholderImage = PLACEHOLDER_IMAGES.coffee;
    } else if (nameLower.includes('weather')) {
        icon = icons.weather;
        placeholderImage = PLACEHOLDER_IMAGES.weather;
    } else if (nameLower.includes('data') || nameLower.includes('analy')) {
        icon = icons.data;
        placeholderImage = PLACEHOLDER_IMAGES.data;
    }

    // Determine image source - use placeholder if local image doesn't exist
    let imageSource = placeholderImage;
    if (project.image) {
        if (project.image.startsWith('http')) {
            imageSource = project.image;
        } else if (fs.existsSync(path.join(__dirname, project.image))) {
            imageSource = project.image;
        }
    }

    const featuredClass = project.featured ? ' featured' : '';

    // Generate tech tags
    const techTags = project.techStack
        .map(tech => `<span class="tech-tag">${tech}</span>`)
        .join('\n                    ');

    // Generate course info
    const courseMatch = project.course.match(/IBM\s+[\w\s]+Course/i);
    const courseText = courseMatch ? courseMatch[0] : project.course.replace(/Completed as (a )?part of (an? )?/i, '');

    // Generate image section
    const imageSection = `
                <div class="project-image">
                    <img src="${imageSource}" alt="${project.name} preview">
                </div>`;

    // Generate action buttons
    let actions = `<a href="${project.url}" class="project-link" target="_blank" rel="noopener noreferrer">
                        View on GitHub
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                            <polyline points="15 3 21 3 21 9"></polyline>
                            <line x1="10" y1="14" x2="21" y2="3"></line>
                        </svg>
                    </a>`;

    if (project.hasMap) {
        actions = `<div class="project-actions">
                    ${actions}
                    <a href="folium_map.html" class="project-link secondary">
                        View Interactive Map
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"></polygon>
                            <line x1="8" y1="2" x2="8" y2="18"></line>
                            <line x1="16" y1="6" x2="16" y2="22"></line>
                        </svg>
                    </a>
                </div>`;
    }

    return `            <article class="project-card${featuredClass}">${imageSection}
                <div class="project-header">
                    <span class="project-icon">
                        ${icon}
                    </span>
                    <h2>${project.name}</h2>
                </div>
                <p class="project-description">${project.description}</p>
                <div class="tech-stack">
                    ${techTags}
                </div>
                <p class="project-course">Completed as part of <strong>${courseText}</strong></p>
                ${actions}
            </article>`;
}

// Generate personal info section HTML
function generatePersonalInfoSection(personalInfo) {
    if (!personalInfo) return '';

    // Use placeholder if image is a local path that may not exist
    let image = personalInfo.image || PLACEHOLDER_IMAGES.profile;
    if (image.startsWith('images/') && !fs.existsSync(path.join(__dirname, image))) {
        image = PLACEHOLDER_IMAGES.profile;
    }

    const detailsHtml = personalInfo.details
        .map(d => {
            if (d.key) {
                return `<li><strong>${d.key}:</strong> ${d.value}</li>`;
            }
            return `<li>${d.value}</li>`;
        })
        .join('\n                        ');

    return `
        <section class="about-section">
            <div class="about-container">
                <div class="about-image">
                    <img src="${image}" alt="${personalInfo.name || 'Profile photo'}">
                </div>
                <div class="about-content">
                    <h2>${personalInfo.name || 'About Me'}</h2>
                    ${personalInfo.bio ? `<p class="about-bio">${personalInfo.bio}</p>` : ''}
                    ${detailsHtml ? `
                    <ul class="about-details">
                        ${detailsHtml}
                    </ul>` : ''}
                </div>
            </div>
        </section>`;
}

// Generate the full HTML
function generateHtml(data) {
    const projectCards = data.projects.map((p, i) => generateProjectCard(p, i)).join('\n\n');
    const personalSection = generatePersonalInfoSection(data.personalInfo);

    // Check if any project has a map
    const hasMapProject = data.projects.some(p => p.hasMap);
    const mapSection = hasMapProject ? `
        <section class="map-preview">
            <h2>Chicago Libraries & Optimal Coffee Shop Locations</h2>
            <p>Interactive map showing library locations and the algorithmically determined optimal positions for coffee shops.</p>
            <div class="map-container">
                <iframe src="folium_map.html" title="Interactive Map of Chicago Libraries and Optimal Coffee Shop Locations"></iframe>
            </div>
        </section>` : '';

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Python Data Science Projects | Rio O'Dwyer</title>
    <link rel="stylesheet" href="styles.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <header class="hero">
        <div class="container">
            <h1>${data.title || 'Python Projects'}</h1>
            <p class="subtitle">${data.description || 'A collection of data science projects.'}</p>
        </div>
    </header>

    <main class="container">${personalSection}

        <section class="projects">
${projectCards}
        </section>
${mapSection}
    </main>

    <footer>
        <div class="container">
            <p>Built with Python & Data Science</p>
            <a href="https://github.com/rioodwyer1611/PythonProjects" target="_blank" rel="noopener noreferrer">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
            </a>
        </div>
    </footer>
</body>
</html>`;
}

// Main execution
const data = parseReadme(readmeContent);
const html = generateHtml(data);

// Write the generated HTML
fs.writeFileSync(path.join(__dirname, 'index.html'), html);

console.log('Successfully generated index.html from README.md');
console.log(`Found ${data.projects.length} projects`);
if (data.personalInfo) {
    console.log('Personal info section detected');
}
