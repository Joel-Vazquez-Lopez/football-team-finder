"""
IMPORTS
"""
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
import random

"""
TEAMS: 


# Vector format:

# Creativity 
# Discipline 
# Emotional
# Aggression 
# Loyalty 
# Glory 
# Risk 
# Structure

"""

teams = {

"Barcelona": [0.95, 0.65, 0.70, 0.40, 0.60, 0.90, 0.85, 0.60],
# Extremely creative, high risk, expressive attacking team

"Real Madrid": [0.75, 0.80, 0.60, 0.60, 0.50, 0.95, 0.75, 0.70],
# Elite, structured winners, high ambition and efficiency

"Villarreal": [0.70, 0.75, 0.60, 0.50, 0.60, 0.75, 0.65, 0.75],
# Balanced, intelligent, tactically solid with some creativity

"Atletico Madrid": [0.35, 0.90, 0.75, 0.95, 0.70, 0.80, 0.40, 0.95],
# Ultra-disciplined, aggressive, defensive "warrior" mentality


"Real Betis": [0.80, 0.55, 0.80, 0.50, 0.75, 0.60, 0.75, 0.50],
# Creative, emotional, entertaining but less structured

"Celta Vigo": [0.70, 0.60, 0.80, 0.50, 0.70, 0.55, 0.65, 0.60],
# Technical, fluid, emotional mid-table style

"Real Sociedad": [0.65, 0.75, 0.55, 0.55, 0.65, 0.60, 0.60, 0.80],
# Structured, tactical, controlled with balanced play


"Athletic Bilbao": [0.60, 0.80, 0.80, 0.70, 0.95, 0.65, 0.55, 0.85],
# Extremely loyal identity, emotional, disciplined and structured

"Sevilla": [0.65, 0.65, 0.75, 0.65, 0.80, 0.70, 0.60, 0.65],
# Emotional, experienced, competitive with strong identity

"Valencia": [0.60, 0.70, 0.80, 0.65, 0.85, 0.65, 0.60, 0.70],
# Passionate, historical club, emotional with solid structure


"Getafe": [0.30, 0.85, 0.60, 0.90, 0.75, 0.40, 0.30, 0.95],
# Very defensive, aggressive, highly structured "anti-football"

"Osasuna": [0.50, 0.80, 0.70, 0.75, 0.90, 0.50, 0.50, 0.85],
# Loyal, hard-working, disciplined underdog team

"Rayo Vallecano": [0.75, 0.50, 0.85, 0.65, 0.85, 0.45, 0.80, 0.45],
# Chaotic, emotional, attacking, high-risk underdog


"Espanyol": [0.55, 0.65, 0.65, 0.60, 0.70, 0.50, 0.55, 0.65],
# Balanced, moderate identity, mid-level everything

"Mallorca": [0.50, 0.70, 0.65, 0.70, 0.80, 0.45, 0.50, 0.75],
# Defensive, aggressive, structured survival-focused

"Alaves": [0.45, 0.75, 0.65, 0.75, 0.85, 0.40, 0.45, 0.80],
# Hard-working, defensive, loyal underdog

"Girona": [0.80, 0.55, 0.75, 0.50, 0.70, 0.60, 0.80, 0.50],
# Very attacking, creative, high-risk emerging team


"Levante": [0.60, 0.50, 0.80, 0.60, 0.85, 0.35, 0.75, 0.45],
# Emotional, attacking underdog, less structured

"Elche": [0.45, 0.60, 0.70, 0.65, 0.85, 0.30, 0.50, 0.65],
# Struggling, loyal, moderately defensive

"Real Oviedo": [0.50, 0.65, 0.85, 0.60, 0.95, 0.30, 0.55, 0.60]
# Extremely loyal, emotional, identity-driven club

}


teams_epl = {

"Manchester City": [0.90, 0.90, 0.50, 0.60, 0.50, 0.95, 0.70, 0.95],
# Hyper-structured, dominant, system perfection, low chaos

"Arsenal": [0.85, 0.70, 0.70, 0.55, 0.70, 0.80, 0.80, 0.60],
# Creative, attacking, expressive but slightly unstable

"Liverpool": [0.80, 0.75, 0.95, 0.85, 0.90, 0.85, 0.85, 0.70],
# Extremely emotional, intense, high energy and loyalty

"Chelsea": [0.75, 0.70, 0.65, 0.65, 0.60, 0.85, 0.75, 0.70],
# Talented, ambitious, but inconsistent identity

"Manchester United": [0.75, 0.65, 0.85, 0.70, 0.85, 0.90, 0.80, 0.60],
# Emotional, historic giant, chaotic but ambitious


"Tottenham": [0.80, 0.55, 0.85, 0.60, 0.75, 0.75, 0.85, 0.50],
# Very attacking, emotional, chaotic, high risk

"Newcastle United": [0.70, 0.80, 0.80, 0.75, 0.85, 0.80, 0.70, 0.80],
# Structured but passionate, rising ambitious force

"Aston Villa": [0.75, 0.75, 0.75, 0.65, 0.80, 0.75, 0.75, 0.75],
# Balanced, structured, ambitious but still identity-driven

"Brighton": [0.90, 0.65, 0.70, 0.50, 0.70, 0.65, 0.90, 0.60],
# Highly creative, risky, intelligent attacking system

"West Ham": [0.60, 0.75, 0.80, 0.70, 0.85, 0.70, 0.60, 0.75],
# Physical, emotional, loyal, structured underdog


"Crystal Palace": [0.70, 0.65, 0.85, 0.65, 0.85, 0.55, 0.70, 0.65],
# Emotional, flair moments, identity-driven mid-table

"Brentford": [0.75, 0.80, 0.65, 0.65, 0.80, 0.65, 0.70, 0.85],
# Smart, data-driven, structured but still attacking

"Fulham": [0.70, 0.70, 0.65, 0.60, 0.75, 0.60, 0.65, 0.70],
# Balanced, controlled, mid-level across all traits

"Wolves": [0.65, 0.80, 0.65, 0.75, 0.80, 0.60, 0.55, 0.85],
# Defensive, aggressive, structured and compact

"Everton": [0.55, 0.75, 0.85, 0.75, 0.90, 0.65, 0.50, 0.80],
# Very emotional, loyal, defensive struggle identity


"Nottingham Forest": [0.65, 0.65, 0.85, 0.70, 0.90, 0.60, 0.65, 0.65],
# Historic, emotional, identity-first, unstable structure

"Burnley": [0.60, 0.70, 0.70, 0.65, 0.80, 0.55, 0.60, 0.70],
# Traditional, disciplined, balanced but limited creativity

"Bournemouth": [0.75, 0.60, 0.70, 0.60, 0.75, 0.55, 0.75, 0.60],
# Open, attacking, slightly chaotic small club

"Sheffield United": [0.50, 0.70, 0.75, 0.75, 0.85, 0.40, 0.50, 0.75],
# Hard-working, aggressive, survival-focused

"Luton Town": [0.55, 0.65, 0.90, 0.75, 0.95, 0.35, 0.60, 0.65]
# Maximum loyalty + emotion, classic underdog story

}


teams_bundesliga = {

"Bayern Munich": [0.85, 0.90, 0.60, 0.70, 0.60, 0.98, 0.75, 0.95],
# Dominant machine, elite, structured, winning mentality

"Borussia Dortmund": [0.90, 0.65, 0.95, 0.80, 0.95, 0.85, 0.95, 0.60],
# Extremely emotional, explosive, high-risk, fan-driven

"RB Leipzig": [0.75, 0.85, 0.55, 0.75, 0.40, 0.80, 0.70, 0.90],
# Modern, system-based, efficient, less emotional identity

"Bayer Leverkusen": [0.90, 0.80, 0.70, 0.65, 0.65, 0.85, 0.90, 0.80],
# Highly creative + structured, attacking but controlled


"Union Berlin": [0.50, 0.85, 0.90, 0.80, 0.98, 0.60, 0.50, 0.90],
# Maximum loyalty, underdog, disciplined, identity-first

"SC Freiburg": [0.65, 0.85, 0.75, 0.70, 0.90, 0.65, 0.60, 0.90],
# Smart, disciplined, community-driven, consistent

"Eintracht Frankfurt": [0.80, 0.65, 0.90, 0.75, 0.90, 0.70, 0.85, 0.65],
# Emotional, attacking, chaotic but passionate

"Wolfsburg": [0.65, 0.80, 0.60, 0.70, 0.65, 0.70, 0.60, 0.85],
# Structured, physical, balanced but less expressive


"Borussia Mönchengladbach": [0.80, 0.65, 0.75, 0.65, 0.80, 0.65, 0.80, 0.60],
# Creative, attacking, but inconsistent identity

"Hoffenheim": [0.75, 0.70, 0.65, 0.65, 0.60, 0.65, 0.75, 0.70],
# Balanced, slightly attacking, modern but not extreme

"Werder Bremen": [0.75, 0.60, 0.85, 0.70, 0.90, 0.65, 0.80, 0.60],
# Emotional, traditional, attacking and unstable

"Mainz 05": [0.60, 0.75, 0.75, 0.75, 0.85, 0.55, 0.65, 0.80],
# Hard-working, aggressive, loyal mid-table


"FC Köln": [0.70, 0.65, 0.95, 0.70, 0.95, 0.60, 0.75, 0.65],
# Extremely emotional, fan-driven, chaotic but lovable

"Augsburg": [0.55, 0.75, 0.70, 0.75, 0.85, 0.50, 0.55, 0.80],
# Defensive, aggressive, survival-focused

"VfB Stuttgart": [0.85, 0.65, 0.80, 0.65, 0.85, 0.70, 0.85, 0.60],
# Creative, attacking, high-risk, exciting rising team


"Hertha Berlin": [0.70, 0.60, 0.80, 0.65, 0.85, 0.65, 0.75, 0.60],
# Emotional, unstable, big-club potential but chaotic

"Schalke 04": [0.60, 0.65, 0.95, 0.75, 0.98, 0.70, 0.65, 0.70],
# Extreme loyalty + emotion, dramatic, historic club

"Bochum": [0.55, 0.70, 0.85, 0.75, 0.95, 0.50, 0.60, 0.70],
# Underdog, emotional, physical, identity-driven

"Heidenheim": [0.60, 0.75, 0.75, 0.70, 0.90, 0.45, 0.65, 0.80],
# Small club, disciplined, loyal, organized

"Darmstadt": [0.55, 0.65, 0.80, 0.70, 0.90, 0.40, 0.60, 0.70]
# Emotional, underdog, less structured, survival mindset

}


teams_inazuma = {

"Raimon": [0.85, 0.75, 0.95, 0.70, 0.98, 0.80, 0.85, 0.70],
# Balanced, growth, friendship-driven, emotional core team

"Royal Academy": [0.60, 0.95, 0.70, 0.85, 0.80, 0.90, 0.60, 0.95],
# Highly disciplined, dominant, authoritarian structure

"Zeus": [0.95, 0.80, 0.60, 0.70, 0.50, 0.98, 0.90, 0.85],
# Elite, flashy, superiority complex, technical dominance

"Royal Academy Redux": [0.65, 0.90, 0.75, 0.80, 0.85, 0.85, 0.65, 0.90],
# Structured but more balanced evolution of Teikoku

"Alius Academy": [0.70, 0.85, 0.65, 0.98, 0.40, 0.95, 0.85, 0.85],
# Overwhelming power, aggressive, almost chaotic domination

"Epsilon": [0.65, 0.90, 0.60, 0.95, 0.35, 0.95, 0.75, 0.90],
# Strong, relentless, system-based destruction

"Genesis": [0.85, 0.85, 0.70, 0.90, 0.50, 0.98, 0.85, 0.90],
# Perfect blend of power, talent, and structure

"Diamond Dust": [0.80, 0.85, 0.65, 0.80, 0.60, 0.90, 0.75, 0.90],
# Cold, calculated, elegant but controlled

"Chaos": [0.85, 0.80, 0.75, 0.90, 0.55, 0.92, 0.85, 0.85],
# Fusion of balance and destruction, unpredictable

"Inazuma Japan": [0.85, 0.80, 0.98, 0.80, 0.98, 0.95, 0.85, 0.80],
# Unity, passion, elite international spirit

"Little Gigant": [0.75, 0.98, 0.70, 0.75, 0.80, 0.98, 0.70, 0.98],
# Perfect structure, intelligence, almost flawless system

"Knights of Queen": [0.85, 0.80, 0.75, 0.75, 0.85, 0.90, 0.80, 0.85],
# Elegant, balanced, refined football

"Fire Dragon": [0.85, 0.75, 0.90, 0.85, 0.90, 0.90, 0.85, 0.75],
# Passionate, aggressive, emotional powerhouse

"Orpheus": [0.95, 0.80, 0.80, 0.70, 0.85, 0.85, 0.90, 0.80],
# Artistic, creative, beautiful playstyle

"The Empire": [0.60, 0.95, 0.75, 0.85, 0.85, 0.90, 0.60, 0.95],
# Structured, traditional, disciplined football nation

"Unicorn": [0.80, 0.75, 0.75, 0.80, 0.85, 0.85, 0.85, 0.75],
# Balanced, strong, emotionally driven

"Big Waves": [0.75, 0.70, 0.80, 0.85, 0.85, 0.80, 0.85, 0.70],
# Energetic, physical, emotional and aggressive

"Desert Lion": [0.70, 0.80, 0.85, 0.85, 0.90, 0.85, 0.75, 0.80],
# Strong identity, pride, resilience

"Neo Japan": [0.80, 0.85, 0.60, 0.80, 0.70, 0.90, 0.70, 0.90],
# Structured, competitive, less emotional

"Team Garshield": [0.60, 0.95, 0.55, 0.85, 0.60, 0.90, 0.60, 0.98]
# Extremely controlled, rigid, system-first team

}

teams_national = {

"Brazil": [0.98, 0.75, 0.90, 0.80, 0.85, 0.98, 0.95, 0.75],
# Maximum creativity, flair, attacking freedom, expressive football

"Argentina": [0.90, 0.80, 0.98, 0.85, 0.95, 0.98, 0.90, 0.80],
# Passion, emotion, individual brilliance + national pride

"Spain": [0.85, 0.98, 0.75, 0.60, 0.85, 0.95, 0.70, 0.98],
# Total control, possession, structure, intelligence

"Germany": [0.75, 0.98, 0.70, 0.85, 0.85, 0.98, 0.70, 0.98],
# Discipline, efficiency, structured dominance

"France": [0.90, 0.85, 0.80, 0.90, 0.80, 0.98, 0.85, 0.85],
# Power + talent, explosive, balanced elite team

"England": [0.80, 0.85, 0.90, 0.90, 0.95, 0.90, 0.80, 0.85],
# Intensity, physicality, emotional fan-driven identity

"Portugal": [0.90, 0.80, 0.85, 0.75, 0.85, 0.92, 0.85, 0.80],
# Technical, creative, individual brilliance with structure

"Netherlands": [0.90, 0.90, 0.75, 0.75, 0.80, 0.90, 0.85, 0.95],
# Tactical intelligence + attacking philosophy

"Italy": [0.70, 0.98, 0.85, 0.80, 0.90, 0.95, 0.65, 0.98],
# Defensive mastery, structure, experience, control

"Belgium": [0.90, 0.80, 0.80, 0.80, 0.75, 0.90, 0.85, 0.80],
# Golden generation, creative + balanced but inconsistent

"Croatia": [0.85, 0.90, 0.95, 0.80, 0.95, 0.92, 0.80, 0.90],
# Technical, emotional, resilient, strong identity

"Uruguay": [0.70, 0.85, 0.95, 0.95, 0.98, 0.90, 0.70, 0.85],
# Aggressive, emotional, warrior mentality, historic pride

"Morocco": [0.75, 0.90, 0.95, 0.85, 0.98, 0.88, 0.75, 0.90],
# Strong defensive structure + passion + unity

"Japan": [0.85, 0.95, 0.75, 0.70, 0.90, 0.85, 0.80, 0.95],
# Highly organized, disciplined, technical and fast

"South Korea": [0.80, 0.85, 0.90, 0.85, 0.95, 0.85, 0.85, 0.85],
# Intense, energetic, emotionally driven, fast-paced

"USA": [0.75, 0.80, 0.85, 0.85, 0.85, 0.85, 0.80, 0.80],
# Athletic, energetic, improving structure and identity

"Mexico": [0.85, 0.75, 0.95, 0.80, 0.95, 0.88, 0.85, 0.75],
# Passionate, attacking, emotional, fan-driven

"Switzerland": [0.75, 0.90, 0.75, 0.80, 0.85, 0.85, 0.70, 0.90],
# Balanced, structured, disciplined and reliable

"Denmark": [0.80, 0.90, 0.90, 0.75, 0.95, 0.88, 0.75, 0.90],
# Strong team spirit, structure, emotional unity

"Sweden": [0.75, 0.90, 0.80, 0.80, 0.90, 0.88, 0.70, 0.90]
# Structured, disciplined, physical, balanced Scandinavian identity

}


"""
QUESTIONS
"""

questions = [

# --- FOOTBALL (12) ---

{
"question": "What kind of football do you enjoy watching the most?",
"options": {
"A": {"text": "Fast, attacking football with lots of chances", "vector": [1,0,0,0,0,0,1,-0.5]},
"B": {"text": "Controlled, tactical football", "vector": [0,1,0,0,0,0.5,0,1]},
"C": {"text": "Emotional games full of intensity", "vector": [0,0,1,0.5,1,0,0,0]},
"D": {"text": "Physical, hard-fought matches", "vector": [0,0.5,0,1,0,0,0,0.5]}
}
},

{
"question": "Which moment feels the best?",
"options": {
"A": {"text": "A beautiful team goal", "vector": [1,0,0.3,0,0,0.3,0.5,0]},
"B": {"text": "A perfectly executed game plan", "vector": [0,1,0,0,0,0.5,0,1]},
"C": {"text": "A last-minute winner", "vector": [0.3,0,1,0.5,0.5,0.5,0.5,0]},
"D": {"text": "A crucial defensive block", "vector": [0,0.5,0,1,0.3,0,0,1]}
}
},

{
"question": "What type of player do you admire most?",
"options": {
"A": {"text": "Creative playmaker", "vector": [1,0,0.5,0,0,0.3,0.5,0]},
"B": {"text": "Intelligent strategist", "vector": [0,1,0,0,0,0.3,0,1]},
"C": {"text": "Passionate leader", "vector": [0.3,0,1,0.5,1,0.3,0.3,0]},
"D": {"text": "Strong warrior", "vector": [0,0.3,0,1,0.5,0,0,0.5]}
}
},

{
"question": "Your ideal team plays...",
"options": {
"A": {"text": "Expressive and attacking football", "vector": [1,0,0.5,0,0,0.3,1,-0.5]},
"B": {"text": "Structured and disciplined football", "vector": [0,1,0,0,0,0.5,0,1]},
"C": {"text": "With emotion and energy", "vector": [0.3,0,1,0.5,1,0.3,0.5,0]},
"D": {"text": "With strength and resilience", "vector": [0,0.5,0,1,0.5,0,0,0.5]}
}
},

{
"question": "How should a team react after conceding a goal?",
"options": {
"A": {"text": "Attack even more", "vector": [0.8,0,0.3,0.3,0,0,1,-0.5]},
"B": {"text": "Stick to the plan", "vector": [0,1,0,0,0,0.3,0,1]},
"C": {"text": "Use the energy from the crowd", "vector": [0.3,0,1,0.5,1,0.3,0.5,0]},
"D": {"text": "Become tougher and harder to break", "vector": [0,0.5,0,1,0.5,0,0,0.8]}
}
},

{
"question": "What kind of match would you choose to watch?",
"options": {
"A": {"text": "End-to-end attacking game", "vector": [1,0,0.3,0.2,0,0.2,1,-0.5]},
"B": {"text": "Chess-like tactical battle", "vector": [0,1,0,0,0,0.4,0,1]},
"C": {"text": "Derby full of passion", "vector": [0.3,0,1,0.5,1,0.2,0.4,0]},
"D": {"text": "Rough, physical contest", "vector": [0,0.4,0,1,0.3,0,0,0.6]}
}
},

{
"question": "What matters most in football?",
"options": {
"A": {"text": "Style of play", "vector": [1,0,0.3,0,0,0.3,0.6,0]},
"B": {"text": "Winning consistently", "vector": [0,1,0,0,0,1,0,1]},
"C": {"text": "Fans and identity", "vector": [0.2,0,1,0.3,1,0.3,0.2,0]},
"D": {"text": "Effort and toughness", "vector": [0,0.4,0,1,0.4,0,0,0.5]}
}
},

{
"question": "Which manager style do you prefer?",
"options": {
"A": {"text": "Gives freedom and creativity", "vector": [1,0,0.5,0.2,0,0.3,0.8,-0.5]},
"B": {"text": "Highly organized and strategic", "vector": [0,1,0,0,0,0.5,0,1]},
"C": {"text": "Motivational and emotional", "vector": [0.3,0,1,0.4,1,0.3,0.4,0]},
"D": {"text": "Demanding and strict", "vector": [0,0.6,0,1,0.3,0,0,0.7]}
}
},

{
"question": "What makes a team dangerous?",
"options": {
"A": {"text": "Unpredictable creativity", "vector": [1,0,0.3,0.2,0,0.4,1,-0.3]},
"B": {"text": "Tactical discipline", "vector": [0,1,0,0,0,0.4,0,1]},
"C": {"text": "Momentum and passion", "vector": [0.3,0,1,0.5,1,0.3,0.5,0]},
"D": {"text": "Physical dominance", "vector": [0,0.4,0,1,0.3,0,0,0.6]}
}
},

{
"question": "In a big match, your team should:",
"options": {
"A": {"text": "Play their own attacking game", "vector": [1,0,0.3,0.2,0,0.3,0.9,-0.4]},
"B": {"text": "Control and manage the game", "vector": [0,1,0,0,0,0.5,0,1]},
"C": {"text": "Feed off the atmosphere", "vector": [0.3,0,1,0.4,1,0.3,0.4,0]},
"D": {"text": "Fight for every ball", "vector": [0,0.4,0,1,0.4,0,0,0.6]}
}
},

{
"question": "What kind of club story do you like most?",
"options": {
"A": {"text": "A team with a strong philosophy", "vector": [1,0.3,0.3,0,0.3,0.5,0.6,0.3]},
"B": {"text": "A dominant, winning machine", "vector": [0,1,0,0.2,0,1,0,1]},
"C": {"text": "A historic club with loyal fans", "vector": [0.2,0,1,0.3,1,0.5,0.3,0]},
"D": {"text": "An underdog that never gives up", "vector": [0,0.4,0,1,0.6,0.2,0.3,0.5]}
}
},

{
"question": "If your team is winning 1-0 late in the game, you prefer they:",
"options": {
"A": {"text": "Go for a second goal", "vector": [0.9,0,0.3,0.3,0,0.3,1,-0.5]},
"B": {"text": "Keep possession and control", "vector": [0,1,0,0,0,0.4,0,1]},
"C": {"text": "Keep pushing with energy", "vector": [0.3,0,1,0.4,1,0.3,0.5,0]},
"D": {"text": "Defend and protect the lead", "vector": [0,0.5,0,1,0.4,0,0,0.8]}
}
},

# --- LIFE (8) ---

{
"question": "What kind of lifestyle fits you best?",
"options": {
"A": {"text": "Creative and spontaneous", "vector": [1,0,0.4,0,0,0.2,0.8,-0.5]},
"B": {"text": "Structured and organized", "vector": [0,1,0,0,0,0.4,0,1]},
"C": {"text": "Emotional and people-driven", "vector": [0.3,0,1,0.3,1,0.3,0.3,0]},
"D": {"text": "Focused and resilient", "vector": [0,0.5,0,1,0.5,0.2,0,0.6]}
}
},

{
"question": "When facing a challenge, you usually:",
"options": {
"A": {"text": "Try a new approach", "vector": [0.8,0,0.3,0.2,0,0.2,1,-0.3]},
"B": {"text": "Stick to a plan", "vector": [0,1,0,0,0,0.3,0,1]},
"C": {"text": "Follow your instincts", "vector": [0.4,0,1,0.3,0.5,0.2,0.5,0]},
"D": {"text": "Push through no matter what", "vector": [0,0.4,0,1,0.4,0,0,0.5]}
}
},

{
"question": "What motivates you the most?",
"options": {
"A": {"text": "Creating something unique", "vector": [1,0,0.4,0,0,0.3,0.7,0]},
"B": {"text": "Achieving success", "vector": [0,1,0,0,0,1,0,1]},
"C": {"text": "Connecting with people", "vector": [0.3,0,1,0.2,1,0.3,0.3,0]},
"D": {"text": "Overcoming difficulties", "vector": [0,0.4,0,1,0.5,0.3,0,0.5]}
}
},

{
"question": "How do you usually make decisions?",
"options": {
"A": {"text": "Based on intuition", "vector": [0.7,0,0.5,0.2,0,0.2,0.7,-0.3]},
"B": {"text": "Based on logic", "vector": [0,1,0,0,0,0.3,0,1]},
"C": {"text": "Based on feelings", "vector": [0.3,0,1,0.2,1,0.2,0.3,0]},
"D": {"text": "Based on determination", "vector": [0,0.5,0,1,0.4,0,0,0.6]}
}
},

{
"question": "What kind of environment do you prefer?",
"options": {
"A": {"text": "Flexible and creative", "vector": [1,0,0.4,0,0,0.2,0.8,-0.4]},
"B": {"text": "Organized and predictable", "vector": [0,1,0,0,0,0.3,0,1]},
"C": {"text": "Warm and social", "vector": [0.3,0,1,0.2,1,0.3,0.3,0]},
"D": {"text": "Competitive and demanding", "vector": [0,0.5,0,1,0.4,0.4,0,0.5]}
}
},

{
"question": "What frustrates you the most?",
"options": {
"A": {"text": "Lack of creativity", "vector": [1,0,0.3,0,0,0.2,0.6,0]},
"B": {"text": "Lack of structure", "vector": [0,1,0,0,0,0.2,0,1]},
"C": {"text": "Lack of emotion", "vector": [0.3,0,1,0.2,1,0.2,0.2,0]},
"D": {"text": "Lack of effort", "vector": [0,0.4,0,1,0.4,0,0,0.4]}
}
},

{
"question": "In a group, you are usually:",
"options": {
"A": {"text": "The ideas person", "vector": [1,0,0.4,0,0,0.2,0.7,0]},
"B": {"text": "The planner", "vector": [0,1,0,0,0,0.3,0,1]},
"C": {"text": "The motivator", "vector": [0.3,0,1,0.3,1,0.3,0.3,0]},
"D": {"text": "The one who pushes others", "vector": [0,0.4,0,1,0.4,0.2,0,0.5]}
}
},

{
"question": "At your core, what matters most to you?",
"options": {
"A": {"text": "Expression", "vector": [1,0,0.4,0,0,0.3,0.8,0]},
"B": {"text": "Control", "vector": [0,1,0,0,0,0.4,0,1]},
"C": {"text": "Connection", "vector": [0.3,0,1,0.2,1,0.3,0.3,0]},
"D": {"text": "Strength", "vector": [0,0.4,0,1,0.4,0.3,0,0.5]}
}
}

]

print("\nChoose mode:")
print("1 - All teams")
print("2 - Select category")

mode = input("Your choice (1/2): ")

all_teams = {
    **teams,
    **teams_epl,
    **teams_bundesliga,
    **teams_inazuma,
    **teams_national
}

categories = {
    "1": teams,
    "2": teams_epl,
    "3": teams_bundesliga,
    "4": teams_national,
    "5": teams_inazuma
}

if mode == "1":
    selected_teams = all_teams
else:
    print("\nChoose category:")
    print("1 - LaLiga")
    print("2 - Premier League")
    print("3 - Bundesliga")
    print("4 - National Teams")
    print("5 - Inazuma Eleven")

    choice = input("Your choice: ")
    selected_teams = categories.get(choice, all_teams)


answers = []

NUM_QUESTIONS = 15  # you can change this (12, 15, 20...)

selected_questions = random.sample(questions, NUM_QUESTIONS)

total = len(selected_questions)

for i, q in enumerate(selected_questions, 1):
    remaining = total - i

    print(f"\n--- Question {i}/{total} ---")
    print(q["question"])
    
    for key, value in q["options"].items():
        print(f"{key}: {value['text']}")
    
    print(f"({remaining} questions left)")
    
    ans = input("Your answer (A/B/C/D): ").upper()
    
    while ans not in ["A", "B", "C", "D"]:
        ans = input("Please enter A, B, C or D: ").upper()
    
    answers.append(ans)

    # progress bar
    progress = int((i / total) * 20)
    bar = "#" * progress + "-" * (20 - progress)

    print(f"[{bar}] {i}/{total}")


"""
Vector Space for user
"""
user = np.zeros(8)

for i, ans in enumerate(answers):
    vec = selected_questions[i]["options"][ans]["vector"]
    user += np.array(vec)
if np.linalg.norm(user) != 0:
    user = user / np.linalg.norm(user)
"""
NOT USEFUL FOR A SIGNIFICANT WORK 

def apply_answer(user, answer):
    if answer == "A":
        user[0] += 1      # creativity
        user[6] += 1      # risk
        user[7] -= 0.5    # less structure

    elif answer == "B":
        user[1] += 1      # discipline
        user[7] += 1      # structure
        user[5] += 0.5    # glory

    elif answer == "C":
        user[2] += 1      # emotional
        user[4] += 1      # loyalty
        user[3] += 0.5    # aggression

    elif answer == "D":
        user[3] += 1      # aggression
        user[1] += 0.5    # discipline
        user[7] += 0.5    # structure

    return user


user = [0]*8

for ans in answers:
    user = apply_answer(user, ans)

user = np.array(user)
user = user / np.linalg.norm(user)

all_teams = {}
all_teams.update(teams)
all_teams.update(teams_epl)
all_teams.update(teams_bundesliga)
"""



results = []

results = sorted(results, key=lambda x: x[1], reverse=True)

print("\nYour top matches:\n")

for team, score in results[:5]:
    print(f"{team}: {round(score*100, 1)}% match")


def explain_match(user, team_vec):
    traits = ["Creativity", "Discipline", "Emotion", "Aggression",
              "Loyalty", "Glory", "Risk", "Structure"]
    
    diffs = [(t, u * tv) for t, u, tv in zip(traits, user, team_vec)]
    top = sorted(diffs, key=lambda x: x[1], reverse=True)[:2]
    
    print("\nWhy this team fits you:")
    for t, _ in top:
        print(f"- Strong match in {t}")


best_team, best_score = results[0]
explain_match(user, np.array(selected_teams[best_team]))

data = []

fav_team = input("\nWhich team do you actually like? (optional): ")

data.append({
    "answers": answers,
    "team": fav_team
})


with open("data.json", "a") as f:
    f.write(json.dumps(data[-1]) + "\n")
