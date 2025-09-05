import re, wordfreq, nltk
from collections import OrderedDict

print('Building comprehensive word list for keyboard predictions...')

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

# Start with frequent words from wordfreq (aim for ~150k base words)
print('Getting frequent words from wordfreq...')
frequent_words = []
for w in wordfreq.top_n_list('en', 280000):  # Get more to account for filtering
    clean = w.replace("'", '').replace('-', '').lower()
    if (re.fullmatch(r'[a-z]{2,25}', clean) and  # 2-25 chars, much more generous
        len(clean) >= 2 and 
        (clean in valid_words or wordfreq.word_frequency(w, 'en') > 5e-7)):
        frequent_words.append(clean)

print(f'Filtered to {len(frequent_words)} frequent dictionary words')

# Extensive internet slang and abbreviations
internet_slang = [
    # Classic internet
    'lol', 'lmao', 'rofl', 'lmfao', 'omg', 'wtf', 'brb', 'ttyl', 'idk', 'imo', 'imho',
    'tbh', 'ngl', 'smh', 'fml', 'yolo', 'fomo', 'jomo', 'goat', 'periodt', 'stfu',
    'cap', 'nocap', 'ig', 'salty', 'stan', 'simp', 'karen', 'chad', 'boomer', 'zoomer',
    'millennial', 'gen', 'genz', 'alpha', 'sigma', 'based', 'cringe', 'sus', 'sussy',
    'baka', 'uwu', 'owo', 'poggers', 'pog', 'copium', 'hopium', 'ratio', 'ratioed',
    
    # Messaging abbreviations  
    'dm', 'dms', 'pm', 'pms', 'msg', 'txt', 'im', 're', 'retweet', 'rt',
    'fav', 'sub', 'share', 'yt',
    'notif', 'notifs', 'ping', 'tg', 'tag', 'goog', 'meta', 'tok',
    'snap', 'snapchat', 'insta', 'reels', 'tiktok', 'twitter', 'fb',
    'youtube', 'twitch', 'discord', 'reddit', 'linkedin', 'whatsapp', 'telegram',
    
    # Modern expressions
    'vibe', 'vibes', 'vibing', 'mood', 'big', 'mood', 'same', 'felt', 'valid',
    'slay', 'slaying', 'queen', 'king', 'icon', 'iconic', 'legend', 'legendary',
    'fire', 'lit', 'lowkey', 'highkey', 'deadass', 'fr', 'frfr', 'ong', 'istg',
    'swear', 'oath', 'word', 'real', 'fax', 'spill', 'tea', 'drag', 'dragging',
    'pressed', 'triggered', 'toxic', 'wholesome', 'cursed', 'blessed', 'mood',
    
    # Reaction words
    'oop', 'oops', 'yikes', 'ope', 'oof', 'rip', 'sadge', 'pepehands', 'kekw', 'omegalul',
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

# Tech and digital terms
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
    'algo', 'ai', 'agi', 'vibecoding', 'ml', 'dnn',
    'deeplearning', 'nn', 'blockchain', 'solana', 'cryptocurrency', 'crypto',
    'bitcoin', 'ethereum', 'nft', 'metaverse', 'vr', 'ar', 'podman', 'dns',
    'ar', 'iot', 'xr', 'dockerfile', 'cgi', 'bot', 'chatbot',
    'api', 'sdk', 'llm', 'llms', 'db', 'fe', 'be',
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
    'terms', 'conditions', 'policy', 'procedure', 'process', 'workflow'
]

# Academic and professional fields
academic_terms = [
    # General academic
    'university', 'college', 'school', 'education', 'degree', 'bachelor',
    'master', 'phd', 'doctorate', 'professor', 'student', 'research', 'thesis',
    'dissertation', 'paper', 'journal', 'conference', 'publication', 'peer',
    'review', 'citation', 'reference', 'bibliography', 'abstract', 'conclusion',
    
    # STEM fields
    'science', 'technology', 'engineering', 'mathematics', 'physics', 'chemistry',
    'biology', 'computer', 'data', 'statistics', 'analysis', 'experiment',
    'hypothesis', 'theory', 'algorithm', 'formula', 'equation', 'variable',
    'function', 'graph', 'chart', 'diagram', 'model', 'simulation', 'coding',
    'programming', 'development', 'software', 'hardware', 'system', 'network'
]

# Medical and health (common terms people type)
health_terms = [
    'doc', 'hospital', 'clinic', 'apt', 'prescription', 'addy',
    'rx', 'scrip', 'insurance', 'health', 'wellness', 'fitness',
    'exercise', 'workout', 'gym', 'nutrition', 'diet', 'calories', 'protein',
    'vitamins', 'supplements', 'therapy', 'treatment', 'surgery', 'recovery',
    'symptoms', 'diagnosis', 'condition', 'disease', 'illness', 'injury',
    'pain', 'headache', 'fever', 'cold', 'flu', 'covid', 'vaccine', 'vaccination',
    'mask', 'sanitizer', 'quarantine', 'isolation', 'pandemic', 'virus', 'bacteria'
]

# Common abbreviations people actually type
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
    
    # Location abbreviations
    'usa', 'uk', 'ca', 'au', 'nyc', 'la', 'sf', 'gr', 'mi', 'chitown', 'miami', 'boston',
    'seattle', 'portland', 'austin', 'dallas', 'houston', 'atlanta', 'denver',
    'vegas', 'orlando', 'tampa', 'phoenix', 'sandiego', 'sacramento', 'detroit'
]

# Combine all additional terms
all_additional = (internet_slang + tech_terms + business_abbrevs + 
                 academic_terms + health_terms + common_abbrevs)

print(f'Added {len(all_additional)} additional terms')

# Combine everything and remove duplicates while preserving frequency order
print('Combining and deduplicating...')
all_words = frequent_words + all_additional
seen = set()
unique_words = []

for word in all_words:
    if word not in seen and len(word) >= 2 and len(word) <= 25:
        seen.add(word)
        unique_words.append(word)
        if len(unique_words) >= 200000:  # Stop at 200k
            break

print(f'Final word list: {len(unique_words)} words')
print(f'Sample words: {unique_words[:20]}')
print(f'Length distribution: 2-4 chars: {len([w for w in unique_words if 2 <= len(w) <= 4])}, 5-8 chars: {len([w for w in unique_words if 5 <= len(w) <= 8])}, 9+ chars: {len([w for w in unique_words if len(w) >= 9])}')

# Output the final list
for word in unique_words:
    print(word)