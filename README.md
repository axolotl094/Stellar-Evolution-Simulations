# Stellar Evolution Simulations

A computational suite modeling Jeans instability, quantum tunneling in stellar fusion, and Type Ib supernova dynamics.

---

###  Watch the Talk
This simulation suite was built for a talk on *Stellar Evolution*.
You can watch the full lecture on YouTube-> https://youtu.be/-i0FiBEmWxA?si=NSn7P7D6QB0ai48j
---

##  Overview

1. Jeans Collapse

File: jeanslimit.py

Every star starts as a cloud of gas. Gravity wants to squeeze it, and the gas pressure wants to push back. The Jeans mass is the tipping point: a cloud heavier than its Jeans mass collapses, and a lighter one just puffs back out.

This simulation builds a spherical cloud out of a few hundred gas particles in 3D, switches on self-gravity and gas pressure, and lets you watch which side wins.

Controls

Cloud mass slider (0.1 to 1000 solar masses)
Density slider (10 to a million particles per cm³)
Restart button

Moving either slider restarts the cloud with the new settings.

What you see

The 3D cloud, coloured by local density
A "Jeans mass diagram" showing the collapse and stable regions, with a dot marking your current cloud
How the cloud's radius changes with time
A live readout of the cloud's mass compared with its Jeans mass (M / M_J) and what it is doing

Things to try

Keep the density fixed and slide the mass across the Jeans mass. Watch the cloud flip from dispersing to collapsing.
Keep the mass fixed and raise the density. The Jeans mass drops, so a cloud that was safe starts to collapse.
Compare the collapse time at low and high density. Denser clouds fall in faster.

The physics in a nutshell

The Jeans mass for an isothermal gas is M_J = (π^(5/2) / 6) · c_s³ / (G^(3/2) · ρ^(1/2)). Cooler or denser gas gives a smaller Jeans mass, which is why stars form in cold, dense molecular clouds.

Honest caveats

The temperature is fixed at 10 K (a typical molecular cloud) and the cloud sits in empty space with no outside pressure, so the boundary between "collapse" and "disperse" is approximate, not razor sharp.
There is no radiative cooling and no mechanism to turn a dense core into a star (no "sink particles"). After a very violent collapse, the dense core can bounce and scatter particles outward. Treat the late-time behaviour of extreme settings as qualitative.
With only a few hundred particles it shows the right behaviour, but fine detail is limited.


2. Quantum Tunneling

File: tunneling.py

At the temperature of the Sun's core, protons do not have nearly enough energy to climb over their mutual electric repulsion. Fusion happens anyway, because quantum particles can tunnel through barriers they classically cannot cross. Without it, stars would not shine.

This simulation sends a quantum wave packet at a barrier and solves the time-dependent Schrödinger equation in 1D, so you can watch the wave split into a reflected part and a transmitted part.

Controls

Barrier height slider
Barrier thickness slider
Restart button

The particle's energy is fixed (the green dashed line), so the sliders change only the barrier.

What you see

The wave function itself, drawn as its real part, its imaginary part, and its amplitude (the dashed envelope)
The barrier as a red block
A live readout of how much of the wave was reflected, how much is inside the barrier, and how much was transmitted
The exact textbook transmission for comparison

Things to try

Make the barrier taller than the particle's energy, then thicken it. Transmission falls off roughly exponentially with thickness.
Look at ψ inside the barrier. It does not oscillate. It decays, which is the signature of tunneling.
Drop the barrier below the particle's energy. The wave now oscillates inside the barrier again and mostly passes through.

Honest caveats

The numerical transmission is close to, but not exactly, the textbook value. The textbook number is for a single energy, while a wave packet contains a spread of energies.
A faint bump can linger over the barrier for a while. It comes from slow, near-barrier-energy components of the packet and fades over time.
The simulation stops at a fixed time, after the packet has passed the barrier, so the final percentages stay on screen.


3. Supernova Type II

File: supernova2.py

A star more than about eight times the mass of the Sun does not fade away quietly. It builds an iron core that cannot release any more energy by fusion, and then it collapses and explodes. What is left at the centre is a neutron star.

This simulation walks through the whole sequence, in six stages:

Pre-supernova: the "onion" star, with hydrogen, helium, carbon, neon/oxygen and silicon burning in shells around an iron core
Collapse: the iron core falls inward at up to roughly a quarter of the speed of light
Bounce and stall: the core stiffens at nuclear density and launches a shock wave, which then stalls
Revival: neutrinos heat the gas behind the shock and push it outward again, while the proto-neutron star cools and shrinks
Shock crossing and breakout: the shock plows through the star, makes nickel-56 on the way, and finally breaks out of the surface
Expansion: the layered, mixed ejecta expand, the neutron star gets a kick, and the glowing photosphere follows the plateau, drop and radioactive tail

Controls

Progenitor mass slider (8 to 25 solar masses)
Explosion energy slider (0.3 to 3 foe, where 1 foe = 10⁵¹ erg)
Ni-56 mass slider (0.005 to 0.3 solar masses)
Restart and Pause buttons

Moving a slider jumps straight to the start of the collapse with the new numbers.

What you see

A cross-section of the star, from the onion-shell structure through the shock and the expanding ejecta
A zoom into the core on a logarithmic radius scale (about 3000 km down to a few kilometres), showing the infalling shells, the shock, the neutrino heating layer and the newborn neutron star
A plot of the shock radius and the proto-neutron-star radius against time after bounce, so you can see the stall and the revival
A plot of the neutrino luminosity, showing the neutronization burst, the accretion phase and the long cooling tail
A running text summary of the stage and the key numbers

Things to try

Raise the explosion energy and watch the shock revive sooner after the stall.
Change the progenitor mass and see how the iron core, the neutron star mass and the burning timescales change. Notice how short the last burning stages are compared with hydrogen burning.
Raise the Ni-56 mass and look at the nickel-rich material that gets mixed outward into the lighter layers.
Pause during the stall and look at the gain layer, where neutrinos deposit the energy that eventually drives the explosion.

The physics in a nutshell

About 99% of the energy of the collapse leaves as neutrinos, and only about 1% becomes the visible explosion. The shock stalls because it loses energy breaking up iron nuclei, and it is the neutrinos, heating the gas behind it, that bring it back to life.

Honest caveats

It is not a hydrodynamics code. The core collapse, the stall and the revival follow analytic fits chosen to match typical literature values, not a full simulation.
Sizes are not to scale. The core panel uses a log radius scale, and the expansion is drawn at a convenient visual size.
The relation between progenitor mass, iron core, neutron star mass and star radius is a simple approximate recipe.
The neutron star kick is exaggerated so that you can see it.
The progenitor mass range stops at 25 solar masses. Much heavier stars can collapse into black holes instead, which this simulation does not show.


AI Attribution & Methodology

Physics & Architecture: Formulated the physical equations (Jeans mass collapse, Gamow factor quantum tunneling, and Type Ib core-collapse dynamics) and simulation parameters.

Code Generation & Prototyping: Used AI coding tools for rapid script scaffolding, Matplotlib plotting setups, and boilerplate code.

Verification: Independently debugged, calibrated, and validated all numerical algorithms against theoretical astrophysics benchmarks.
