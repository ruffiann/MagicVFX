# MagicVFX
MagicVFX: Visual Effects Synthesis in Just Minutes

## 😁Introduction

Visual effects synthesis is crucial in the film and television industry, which aims at enhancing raw footage with virtual elements for greater expressiveness. As the demand for detailed and realistic effects escalates in modern production, professionals are compelled to allocate substantial time and resources to this endeavor. Thus, there is an urgent need to explore more convenient and less resource-intensive methods, such as incorporating the burgeoning Artificial Intelligence Generated Content (AIGC) technology. However, research into this potential integration has yet to be conducted. As the first work to establish a connection between visual effects synthesis and AIGC technology, we start by carefully setting up two paradigms according to the need for pre-produced effects or not: synthesis with reference effects and synthesis without reference effects. Following this, we compile a dataset by processing a collection of effects videos and scene videos, which contains a wide variety of effect categories and scenarios, adequately covering the common effects seen in films and television industry Furthermore, we explore the capabilities of a pre-trained text-to-video model to synthesize visual effects within these two paradigms. The experimental results demonstrate that the pipeline we established can effectively produce impressive visual effects synthesis outcomes, thereby evidencing the significant potential of existing AIGC technology for application in visual effects synthesis tasks.

## 🔆Ours Results

<table>
  <tr>
    <th>input</th>
    <th>Ours</th>
  </tr>
  <tr>
    <td><img src="/gif/AstronautsEarth1.gif" alt="AstronautsEarth1.gif" style="width:100%"></td>
    <td><img src="/gif/AstronautsEarth1-aerolite.gif" alt="AstronautsEarth1-aerolite.gif" style="width:100%"></td>
  </tr>
  <tr>
    <td><img src="/gif/AstronautsSoil.gif" alt="AstronautsSoil.gif" style="width:100%"></td>
    <td><img src="/gif/AstronautsSoil-lightingmiddle.gif" alt="AstronautsSoil-lightingmiddle.gif" style="width:100%"></td>
  </tr>
  <tr>
    <td><img src="/gif/DuskCity.gif" alt="DuskCity.gif" style="width:100%"></td>
    <td><img src="/gif/duskcity-lightinggreen.gif" alt="duskcity-lightinggreen.gif" style="width:100%"></td>
  </tr>
  <tr>
    <td><img src="/gif/WomanSea.gif" alt="WomanSea.gif" style="width:100%"></td>
    <td><img src="/gif/womansea-waterportal.gif" alt="womansea-waterportal.gif" style="width:100%"></td>
  </tr>
  <tr>
    <td><img src="/gif/SpaceDark.gif" alt="SpaceDark.gif" style="width:100%"></td>
    <td><img src="/gif/spacedark-firewarkcircle.gif" alt="spacedark-firewarkcircle.gif" style="width:100%"></td>
  </tr>
</table>

