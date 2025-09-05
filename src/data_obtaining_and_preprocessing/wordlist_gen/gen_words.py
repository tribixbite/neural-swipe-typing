import re, wordfreq, nltk

# Get NLTK words for validation
try:
    from nltk.corpus import words
    valid_words = set(w.lower() for w in words.words())
except:
    print('Downloading NLTK words corpus...')
    nltk.download('words', quiet=True)
    from nltk.corpus import words
    valid_words = set(w.lower() for w in words.words())

print(f'Loaded {len(valid_words)} dictionary words for validation')

# Start with frequent words from wordfreq - moderately permissive
print('Getting frequent words from wordfreq...')
frequent_words = []
for w in wordfreq.top_n_list('en', 400000):  # Reasonable amount
    clean = w.replace("'", '').replace('-', '').lower()
    if (re.fullmatch(r'[a-z]{2,20}', clean) and  
        len(clean) >= 2 and 
        # More reasonable thresholds
        (clean in valid_words or wordfreq.word_frequency(w, 'en') > 5e-8)):
        frequent_words.append(clean)

print(f'Got {len(frequent_words)} frequent words')

# Your excellent additional terms (keeping all your improvements)
internet_slang = [
    # Classic internet
    'lol', 'lmao', 'rofl', 'lmfao', 'omg', 'wtf', 'brb', 'ttyl', 'idk', 'imo', 'imho',
    'tbh', 'ngl', 'smh', 'fml', 'yolo', 'fomo', 'jomo', 'goat', 'periodt', 'stfu',
    'cap', 'nocap', 'ig', 'salty', 'stan', 'simp', 'karen', 'chad', 'boomer', 'zoomer',
    'millennial', 'gen', 'genz', 'alpha', 'sigma', 'based', 'cringe', 'sus', 'sussy',
    'baka', 'uwu', 'owo', 'poggers', 'pog', 'copium', 'hopium', 'ratio', 'ratioed',
    # Messaging abbreviations  
    'dm', 'dms', 'pm', 'pms', 'msg', 'txt', 'im', 're', 'retweet', 'rt',
    'fav', 'sub', 'share', 'yt', 'notif', 'notifs', 'ping', 'tg', 'tag', 'goog', 'meta', 'tok',
    'snap', 'snapchat', 'insta', 'reels', 'tiktok', 'twitter', 'fb',
    'youtube', 'twitch', 'discord', 'reddit', 'linkedin', 'whatsapp', 'telegram',
    # Modern expressions
    'vibe', 'vibes', 'vibing', 'mood', 'big', 'mood', 'same', 'felt', 'valid',
    'slay', 'slaying', 'queen', 'king', 'icon', 'iconic', 'legend', 'legendary',
    'fire', 'lit', 'lowkey', 'highkey', 'deadass', 'fr', 'frfr', 'ong', 'istg',
    'swear', 'oath', 'word', 'real', 'fax', 'spill', 'tea', 'drag', 'dragging',
    'pressed', 'triggered', 'toxic', 'wholesome', 'cursed', 'blessed', 'mood',
    # Reaction words
    'oop', 'wut', 'oops', 'yikes', 'ope', 'oof', 'rip', 'sadge', 'pepehands', 'kekw', 'omegalul',
    'lul', 'kappa', 'sadkek', 'monkas', 'peepo', 'pepe', 'doge', 'stonks', 'hodl',
    'diamond', 'hands', 'paper', 'rocket', 'moon', 'lambo', 'tendies', 'ape', 'apes',
    'eth', 'sol', 'sui', 'gm', 'buidl', 'wen', 'fud',
    # Gaming
    'noob', 'newb', 'pro', 'tryhard', 'sweat', 'sweaty', 'casual', 'hardcore',
    'speedrun', 'speedrunning', 'glitch', 'exploit', 'hack', 'hacks', 'aimbot',
    'wallhack', 'cheat', 'cheater', 'hacker', 'griefing', 'trolling', 'troll',
    'pwned', 'rekt', 'owned', 'dunked', 'clapped', 'diff', 'gap', 'skill', 'issue',
    'git', 'gud', 'ez', 'gg', 'gj', 'wp', 'bg', 'nt', 'ns', 'clutch', 'ace',
    'pentakill', 'quadra', 'triple', 'double', 'solo', 'squad', 'duo', 'trio',
    'lobby', 'queue', 'ranked', 'casual', 'comp', 'competitive', 'scrims', 'scrim'
]

# Tech and digital terms (your improved list)
tech_terms = [
    # Devices and hardware
    'smartphone', 'iphone', 'android', 'tablet', 'ipad', 'laptop', 'desktop',
    'ios', 'pc', 'mac', 'macbook', 'chromebook', 'smartwatch', 'airpods',
    'earbuds', 'iwatch', 'bluetooth', 'wifi', 'ethernet', 'usb', 'hdmi',
    'charger', 'cable', 'adapter', 'dongle', 'hub', 'cri', 'crt', 'kb',
    'mouse', 'trackpad', 'touchscreen', 'usbc', 'ryzen', 'camera', 'webcam',
    'ubiquiti', 'speaker', 'router', 'unifi', 'server', 'cloud', 'cpuz',
    'hdd', 'ssd', 'ram', 'cpu', 'gpu', 'processor', 'graphics', 'card',
    # Software and apps
    'app', 'apps', 'software', 'program', 'application', 'website', 'site',
    'browser', 'chrome', 'firefox', 'safari', 'edge', 'internet', 'explorer',
    'online', 'offline', 'download', 'upload', 'install', 'uninstall', 'update',
    'upgrade', 'patch', 'bug', 'feature', 'beta', 'alpha', 'version', 'release',
    'launch', 'startup', 'shutdown', 'restart', 'reboot', 'crash', 'freeze',
    'lag', 'latency', 'ping', 'bandwidth', 'connection', 'network', 'server',
    # Internet and web
    'email', 'gmail', 'outlook', 'yahoo', 'hotmail', 'inbox', 'spam', 'folder',
    'attachment', 'link', 'url', 'domain', 'subdomain', 'website', 'webpage',
    'homepage', 'login', 'logout', 'signin', 'signout', 'signup', 'register',
    'username', 'password', 'account', 'profile', 'settings', 'preferences',
    'dashboard', 'menu', 'search', 'filter', 'sort', 'pagination', 'scroll',
    'click', 'tap', 'swipe', 'drag', 'drop', 'zoom', 'pinch', 'rotate',
    # Modern tech concepts
    'algo', 'ai', 'agi', 'vibecoding', 'ml', 'dnn', 'deeplearning', 'nn', 
    'blockchain', 'solana', 'cryptocurrency', 'crypto', 'bitcoin', 'ethereum', 
    'nft', 'metaverse', 'vr', 'ar', 'podman', 'dns', 'iot', 'xr', 'dockerfile', 
    'cgi', 'bot', 'chatbot', 'api', 'sdk', 'llm', 'llms', 'db', 'fe', 'be',
    'fullstack', 'devops', 'cicd', 'nextjs', 'vite', 'domain', 'ssl',
    # Streaming and content
    'stream', 'streaming', 'livestream', 'podcast', 'youtube', 'netflix',
    'spotify', 'twitch', 'tiktok', 'instagram', 'facebook', 'twitter', 'reddit',
    'discord', 'zoom', 'teams', 'slack', 'skype', 'facetime', 'whatsapp',
    'telegram', 'signal', 'snapchat', 'pinterest', 'linkedin', 'github',
    'stackoverflow', 'wiki', 'google', 'ama', 'apple', 'microsoft'
]


# Business and professional abbreviations
business_abbrevs = [
    # Common business
    'ceo', 'cto', 'cfo', 'coo', 'vp', 'svp', 'evp', 'director', 'manager',
    'lead', 'senior', 'junior', 'intern', 'contractor', 'consultant', 'freelance',
    'remote', 'wfh', 'office', 'meeting', 'zoom', 'teams', 'slack', 'email',
    'presentation', 'deck', 'slides', 'proposal', 'contract', 'agreement',
    'budget', 'forecast', 'revenue', 'profit', 'loss', 'margin', 'roi', 'kpi',
    'metrics', 'analytics', 'data', 'report', 'dashboard', 'spreadsheet',
    
    # Project management
    'project', 'task', 'milestone', 'deadline', 'timeline', 'schedule', 'sprint',
    'scrum', 'agile', 'kanban', 'jira', 'trello', 'asana', 'notion', 'monday',
    'roadmap', 'backlog', 'story', 'epic', 'feature', 'requirement', 'spec',
    'documentation', 'wiki', 'confluence', 'sharepoint', 'drive', 'dropbox',
    
    # Finance and legal
    'llc', 'inc', 'corp', 'ltd', 'ipo', 'vc', 'pe', 'equity', 'shares', 'stock',
    'options', 'vesting', 'bonus', 'salary', 'compensation', 'benefits', 'pto',
    'vacation', 'sick', 'leave', 'hr', 'legal', 'compliance', 'gdpr', 'privacy',
    'terms', 'conditions', 'policy', 'proc', 'process', 'workflow'
]

# Academic and professional fields
academic_terms = [
    # General academic
    'uni', 'college', 'school', 'edu', 'degree', 'bachelor',
    'master', 'phd', 'doctorate', 'professor', 'student', 'research', 'thesis',
    'dissertation', 'paper', 'journal', 'conference', 'pub', 'peer',
    'review', 'citation', 'ref', 'bib', 'abstract', 'conclusion',
    
    # STEM fields
    'science', 'technology', 'engineering', 'mathematics', 'physics', 'chemistry',
    'biology', 'computer', 'data', 'statistics', 'analysis', 'experiment',
    'hypothesis', 'theory', 'algorithm', 'formula', 'equation', 'variable',
    'function', 'graph', 'chart', 'diagram', 'model', 'simulation', 'coding',
    'programming', 'development', 'software', 'hardware', 'system', 'network'
]

# Medical and health (your improved list)
health_terms = [
    'doc', 'hospital', 'clinic', 'apt', 'prescription', 'addy',
    'rx', 'scrip', 'insurance', 'health', 'wellness', 'fitness',
    'exercise', 'workout', 'gym', 'nutrition', 'diet', 'calories', 'protein',
    'vitamins', 'supplements', 'therapy', 'treatment', 'surgery', 'recovery',
    'symptoms', 'diagnosis', 'condition', 'disease', 'illness', 'injury',
    'pain', 'headache', 'fever', 'cold', 'flu', 'covid', 'vaccine', 'vaccination',
    'mask', 'sanitizer', 'quarantine', 'isolation', 'pandemic', 'virus', 'bacteria'
]

# Common abbreviations (your improved list)
common_abbrevs = [
    # Time and dates
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'am', 'pm', 'est', 'pst', 'cst', 'mst',
    'gmt', 'utc', 'timezone', 'dst', 'today', 'tomorrow', 'yesterday', 'weekend', 'weekday',
    
    # Measurements and units
    'kg', 'lb', 'lbs', 'oz', 'gram', 'grams', 'meter', 'meters', 'cm', 'mm', 'km',
    'inch', 'inches', 'foot', 'feet', 'yard', 'yards', 'mile', 'miles', 'mph', 'kph',
    'celsius', 'fahrenheit', 'temp', 'temperature', 'degree', 'degrees',
    
    # Common shortcuts people type
    'vs', 'aka', 'fyi', 'btw', 'asap', 'eta', 'tbd', 'tba', 'rsvp', 'vip', 'diy',
    'faq', 'howto', 'tutorial', 'guide', 'tips', 'tricks', 'hacks', 'lifehacks',
    'pro', 'tips', 'protips', 'advice', 'help', 'support', 'contact', 'info',
    
    # Location abbreviations (your improved list)
    'usa', 'uk', 'ca', 'au', 'nyc', 'la', 'sf', 'gr', 'mi', 'chitown', 'miami', 'boston',
    'seattle', 'portland', 'austin', 'dallas', 'houston', 'atlanta', 'denver',
    'vegas', 'orlando', 'tampa', 'phoenix', 'sandiego', 'sacramento', 'detroit'
]

# Add even MORE word categories to reach 200k
programming_terms = [
    'proxmox', 'python', 'javascript', 'java', 'cpp', 'csharp', 'php', 'ruby', 'swift', 'kotlin',
    'golang', 'rust', 'typescript', 'html', 'css', 'react', 'vue', 'angular', 'node',
    'django', 'flask', 'rails', 'spring', 'express', 'mongodb', 'mysql', 'postgres',
    'redis', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'firebase', 'vercel',
    'netlify', 'heroku', 'digital', 'ocean', 'github', 'gitlab', 'bitbucket', 'npm',
    'yarn', 'pip', 'maven', 'gradle', 'webpack', 'babel', 'eslint', 'prettier',
    'jest', 'cypress', 'selenium', 'postman', 'insomnia', 'vscode', 'pycharm',
    'intellij', 'sublime', 'atom', 'vim', 'emacs', 'terminal', 'bash', 'zsh',
    'powershell', 'cmd', 'git', 'svn', 'mercurial', 'ci', 'cd', 'agile', 'scrum'
]

food_terms = [
    'kombucha', 'keto', 'paleo', 'gmo', 'msg', 'pfas'
]

entertainment_terms = [
    'botw', 'netflixandchill', 'totk', 'ep', 'lotr', 'snes'
]

# Combine all additional terms
additional_terms = (internet_slang + tech_terms + business_abbrevs + 
                 academic_terms + health_terms + common_abbrevs +
                 programming_terms + food_terms + entertainment_terms)


# EFFICIENT deduplication and combination
print('Efficiently combining and deduplicating...')

# Convert to sets for fast operations
frequent_set = set(frequent_words)

# Remove duplicates from additional terms and filter out ones already in frequent_words
additional_set = set(w for w in additional_terms if w not in frequent_set and 2 <= len(w) <= 25)

print(f'Frequent words: {len(frequent_set)}')
print(f'Additional unique terms: {len(additional_set)}')

# If we need more words, add some more from wordfreq
total_so_far = len(frequent_set) + len(additional_set)
if total_so_far < 200000:
    print(f'Need {200000 - total_so_far} more words, adding from deeper wordfreq...')
    extra_words = []
    for w in wordfreq.top_n_list('en', 500000)[len(frequent_words):]:  # Skip ones we already processed
        clean = w.replace("'", '').replace('-', '').lower()
        if (re.fullmatch(r'[a-z]{2,25}', clean) and 
            clean not in frequent_set and 
            clean not in additional_set and
            # Slightly more permissive for extra words
            (clean in valid_words or wordfreq.word_frequency(w, 'en') > 5e-7)):
            extra_words.append(clean)
            if len(frequent_set) + len(additional_set) + len(extra_words) >= 200000:
                break
    additional_set.update(extra_words)
    print(f'Added {len(extra_words)} extra words')

# Combine efficiently - preserve frequency order
final_words = frequent_words + list(additional_set)[:200000 - len(frequent_words)]

print(f'Final word list: {len(final_words)} words')
print(f'Sample words: {final_words[:20]}')
print(f'Length distribution: 2-4 chars: {len([w for w in final_words if 2 <= len(w) <= 4])}, 5-8 chars: {len([w for w in final_words if 5 <= len(w) <= 8])}, 9+ chars: {len([w for w in final_words if len(w) >= 9])}')

# Output efficiently
for word in final_words:
    print(word)